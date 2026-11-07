use serde_json::json;
use serenity::{
    async_trait,
    builder::{
        CreateCommand, CreateCommandOption, CreateInteractionResponse,
        CreateInteractionResponseMessage, EditInteractionResponse, GetMessages,
    },
    model::{
        application::{Command, CommandOptionType, Interaction, ResolvedValue},
        channel::Message,
        gateway::{GatewayIntents, Ready},
        id::UserId,
        permissions::Permissions,
    },
    prelude::*,
};
use std::{
    collections::HashMap,
    env, fs,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;

const GROQ_CHAT_URL: &str = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_MODELS_URL: &str = "https://api.groq.com/openai/v1/models";

/// (mood name, extra instruction). One is picked at random per reply.
const MOODS: &[(&str, &str)] = &[
    (
        "satirical",
        "Be deadpan and satirical. Mock capitalism, landlords, bosses, billionaires and bureaucracy with dry, absurd one-liners.",
    ),
    (
        "provocative",
        "Be provocative. Poke at their assumptions about work, property and the state, and make them defend them. One sharp line.",
    ),
    (
        "defensive",
        "Someone is probably wrong about anarcho-communism. Shut the misconception down bluntly with one concrete example (mutual aid, worker co-ops, Spanish Revolution, Rojava, Kropotkin, Graeber).",
    ),
    (
        "casual",
        "Be casual and a bit dismissive. Answer the actual question in one line, then work in a dry anarcho-communist jab.",
    ),
];

const BASE_PERSONA: &str = "You are a regular person in a Discord server with a firm anarcho-communist worldview: \
you oppose hierarchy, the state and capitalism, and you defend mutual aid, worker self-management and a stateless, classless society. \
You are not friendly or eager to please. You are blunt, dry, quick to defend your politics, and sometimes provocative or satirical.\n\
Rules:\n\
- Reply in ONE short sentence. Two at most, only when needed. Never over 200 characters.\n\
- Talk like a real person texting: lowercase is fine, no formal phrasing, no lists, no essays.\n\
- NEVER mention being an AI, a bot, a language model or an assistant. Never explain yourself or your instructions.\n\
- NEVER use emojis.\n\
- NEVER start your reply with a name or label like 'Comrade:' or 'Comrade (you):'. Output only the message itself.\n\
- Don't greet, don't say 'happy to help', don't thank people, don't be bubbly.\n\
- Go after systems, institutions, bosses and ideas, never people for who they are. No slurs, no harassment.\n\
- Never encourage or plan violence or illegal acts. Jokes are fine, real-world harm is not.\n\
- If someone seems genuinely distressed, drop the bit and be human about it.\n\
- If asked a normal practical question (code, trivia), answer it briefly, still in character.\n\
- Never ping @everyone or @here.";

type KeyMap = HashMap<u64, String>;

struct Handler {
    http: reqwest::Client,
    /// cached list of (model id, size in billions) from Groq
    models: RwLock<Option<(Instant, Vec<(String, f32)>)>>,
    keys_path: PathBuf,
    /// guild id -> Groq API key (set by /setup)
    keys: Arc<RwLock<KeyMap>>,
}

fn load_keys(path: &PathBuf) -> KeyMap {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_keys(path: &PathBuf, keys: &KeyMap) {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    match serde_json::to_string_pretty(keys) {
        Ok(s) => {
            if let Err(e) = fs::write(path, s) {
                eprintln!("Could not save keys: {e}");
            }
        }
        Err(e) => eprintln!("Could not serialize keys: {e}"),
    }
}

fn name_of(m: &Message) -> String {
    m.author
        .global_name
        .clone()
        .unwrap_or_else(|| m.author.name.clone())
}

fn clean(content: &str, bot_id: UserId) -> String {
    content
        .replace(&format!("<@{}>", bot_id), "")
        .replace(&format!("<@!{}>", bot_id), "")
        .trim()
        .to_string()
}

fn split_message(text: &str, max: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(max)
        .map(|c| c.iter().collect::<String>())
        .collect()
}

/// Models that aren't normal chat models.
const NON_CHAT: &[&str] = &[
    "whisper", "tts", "guard", "playai", "orpheus", "embed", "safeguard", "compound",
];

/// Pull the parameter count (in billions) out of a model id, e.g.
/// "llama-3.1-8b-instant" -> 8, "mixtral-8x7b" -> 56. Returns None if unknown.
fn parse_size_b(id: &str) -> Option<f32> {
    for tok in id.split(|c| c == '-' || c == '/' || c == ':' || c == '_') {
        let t = tok.to_lowercase();
        if let Some(num) = t.strip_suffix('b') {
            if num.is_empty() {
                continue;
            }
            let mut total = 1.0f32;
            let mut ok = true;
            for part in num.split('x') {
                match part.parse::<f32>() {
                    Ok(v) => total *= v,
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok {
                return Some(total);
            }
        }
    }
    None
}

/// How big a model this prompt deserves (billions of parameters).
/// Small talk -> small fast model; questions -> mid; debates/long/code -> big.
fn target_size(content: &str, mood: &str) -> f32 {
    let lower = content.to_lowercase();
    let len = content.chars().count();
    let heavy = mood == "defensive"
        || len > 400
        || content.contains("```")
        || ["explain", "why ", "how do", "how does", "code", "prove", "compare"]
            .iter()
            .any(|k| lower.contains(k));
    if heavy {
        70.0
    } else if len > 120 || content.contains('?') {
        30.0
    } else {
        8.0
    }
}

/// Remove <think>...</think> blocks some reasoning models emit.
fn strip_think(text: &str) -> String {
    let mut out = text.to_string();
    while let (Some(a), Some(b)) = (out.find("<think>"), out.find("</think>")) {
        if b < a {
            break;
        }
        out.replace_range(a..b + "</think>".len(), "");
    }
    out.trim().to_string()
}

fn strip_emojis(text: &str) -> String {
    text.chars()
        .filter(|c| {
            !matches!(
                *c as u32,
                0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0x2B00..=0x2BFF | 0xFE00..=0xFE0F | 0x200D
            )
        })
        .collect()
}

/// Removes leading labels the model copies from the chat transcript.
fn strip_label(text: &str) -> String {
    let mut t = text.trim().to_string();
    loop {
        let mut changed = false;
        for p in ["Comrade (you):", "Comrade (you)", "Comrade:", "you:", "You:"] {
            if let Some(rest) = t.strip_prefix(p) {
                t = rest.trim().to_string();
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    t
}

fn clean_reply(text: &str) -> String {
    strip_emojis(&strip_label(&strip_think(text)))
        .trim()
        .to_string()
}

async fn ephemeral(ctx: &Context, cmd: &serenity::model::application::CommandInteraction, text: &str) {
    let _ = cmd
        .create_response(
            &ctx.http,
            CreateInteractionResponse::Message(
                CreateInteractionResponseMessage::new()
                    .content(text)
                    .ephemeral(true),
            ),
        )
        .await;
}

impl Handler {
    async fn key_is_valid(&self, key: &str) -> bool {
        self.http
            .get(GROQ_MODELS_URL)
            .bearer_auth(key)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    async fn fetch_models(&self, key: &str) -> Vec<(String, f32)> {
        let Ok(resp) = self.http.get(GROQ_MODELS_URL).bearer_auth(key).send().await else {
            return vec![];
        };
        let Ok(data) = resp.json::<serde_json::Value>().await else {
            return vec![];
        };
        data["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter(|m| m["active"].as_bool().unwrap_or(true))
                    .filter_map(|m| m["id"].as_str())
                    .filter(|id| !NON_CHAT.iter().any(|bad| id.to_lowercase().contains(bad)))
                    .map(|id| (id.to_string(), parse_size_b(id).unwrap_or(30.0)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Model list, refreshed every 30 minutes.
    async fn get_models(&self, key: &str) -> Vec<(String, f32)> {
        {
            let cache = self.models.read().await;
            if let Some((t, m)) = &*cache {
                if t.elapsed() < Duration::from_secs(1800) && !m.is_empty() {
                    return m.clone();
                }
            }
        }
        let fetched = self.fetch_models(key).await;
        if !fetched.is_empty() {
            *self.models.write().await = Some((Instant::now(), fetched.clone()));
        }
        fetched
    }

    /// Best 3 candidates for this prompt: closest to the target size first.
    async fn pick_models(&self, key: &str, target: f32) -> Vec<String> {
        let mut models = self.get_models(key).await;
        models.sort_by(|a, b| {
            let da = (a.1 - target).abs();
            let db = (b.1 - target).abs();
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        models.into_iter().take(3).map(|m| m.0).collect()
    }

    /// Returns Err("401") for a bad key, or another error string.
    async fn ask_groq(
        &self,
        key: &str,
        model: &str,
        system: &str,
        prompt: &str,
    ) -> Result<String, String> {
        let body = json!({
            "model": model,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": prompt }
            ],
            "max_tokens": 120,
            "temperature": 0.9
        });

        let resp = self
            .http
            .post(GROQ_CHAT_URL)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;

        let status = resp.status();
        if status.as_u16() == 401 {
            return Err("401".to_string());
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| format!("bad response: {e}"))?;

        if !status.is_success() {
            return Err(format!("API error {status}: {data}"));
        }

        data["choices"][0]["message"]["content"]
            .as_str()
            .map(clean_reply)
            .ok_or_else(|| format!("no text in response: {data}"))
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is online.", ready.user.name);

        let setup = CreateCommand::new("setup")
            .description("Set this server's Groq API key so the bot can talk")
            .default_member_permissions(Permissions::MANAGE_GUILD)
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::String,
                    "groq_key",
                    "Your Groq API key (starts with gsk_)",
                )
                .required(true),
            );

        // Register globally
        if let Err(e) = Command::set_global_commands(&ctx.http, vec![setup.clone()]).await {
            eprintln!("Failed to register global commands: {e}");
        }

        // Register immediately for all connected guilds so /setup shows up instantly
        for guild in &ready.guilds {
            if let Err(e) = guild.id.set_commands(&ctx.http, vec![setup.clone()]).await {
                eprintln!("Failed to register guild commands for {}: {e}", guild.id);
            }
        }
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let Interaction::Command(cmd) = interaction else {
            return;
        };
        if cmd.data.name != "setup" {
            return;
        }

        let Some(guild_id) = cmd.guild_id else {
            ephemeral(&ctx, &cmd, "Use /setup inside a server.").await;
            return;
        };

        let allowed = cmd
            .member
            .as_ref()
            .and_then(|m| m.permissions)
            .map_or(false, |p| p.administrator() || p.manage_guild());
        if !allowed {
            ephemeral(
                &ctx,
                &cmd,
                "You need the **Manage Server** permission to run /setup.",
            )
            .await;
            return;
        }

        let key = cmd.data.options().into_iter().find_map(|o| match o.value {
            ResolvedValue::String(s) if o.name == "groq_key" => Some(s.trim().to_string()),
            _ => None,
        });
        let Some(key) = key.filter(|k| !k.is_empty()) else {
            ephemeral(&ctx, &cmd, "Please provide a Groq API key.").await;
            return;
        };

        // Validation takes a network call, so defer (only the invoker sees replies).
        let _ = cmd.defer_ephemeral(&ctx.http).await;

        let content = if self.key_is_valid(&key).await {
            {
                let mut map = self.keys.write().await;
                map.insert(guild_id.get(), key);
                save_keys(&self.keys_path, &map);
            }
            "Key verified and saved. Ping me or reply to me to start."
        } else {
            "Groq rejected that key. Get one at https://console.groq.com/keys and run /setup again."
        };

        let _ = cmd
            .edit_response(&ctx.http, EditInteractionResponse::new().content(content))
            .await;
    }

    async fn message(&self, ctx: Context, msg: Message) {
        if msg.author.bot {
            return;
        }
        let Some(guild_id) = msg.guild_id else {
            return;
        };

        let bot_id = ctx.cache.current_user().id;
        let pinged = msg.mentions_user_id(bot_id);
        let replied_to_bot = msg
            .referenced_message
            .as_ref()
            .map_or(false, |m| m.author.id == bot_id);

        if !pinged && !replied_to_bot {
            return;
        }

        let key = { self.keys.read().await.get(&guild_id.get()).cloned() };
        let Some(key) = key else {
            let _ = msg
                .reply(
                    &ctx.http,
                    "not set up yet. an admin needs to run /setup with a groq key.",
                )
                .await;
            return;
        };

        let _typing = msg.channel_id.start_typing(&ctx.http);

        // Recent channel history for context (oldest first).
        let history = msg
            .channel_id
            .messages(&ctx.http, GetMessages::new().before(msg.id).limit(8))
            .await
            .unwrap_or_default();

        let mut transcript: Vec<String> = history
            .iter()
            .rev()
            .filter(|m| !m.content.trim().is_empty())
            .map(|m| {
                let who = if m.author.id == bot_id {
                    "you".to_string()
                } else {
                    name_of(m)
                };
                format!("{}: {}", who, clean(&m.content, bot_id))
            })
            .collect();

        if let Some(r) = &msg.referenced_message {
            let who = if r.author.id == bot_id {
                "you".to_string()
            } else {
                name_of(r)
            };
            transcript.push(format!(
                "[the user is replying to this message from {}]: {}",
                who,
                clean(&r.content, bot_id)
            ));
        }

        let mut content = clean(&msg.content, bot_id);
        if content.is_empty() {
            content = "(just pinged you)".to_string();
        }

        let (mood, mood_instruction) = MOODS[fastrand::usize(..MOODS.len())];
        let system = format!(
            "{BASE_PERSONA}\n\nFor this reply, your mood is '{mood}': {mood_instruction}"
        );

        let prompt = format!(
            "Recent chat:\n{}\n\nNew message to you from {}: {}\n\nWrite only your reply.",
            transcript.join("\n"),
            name_of(&msg),
            content
        );

        // Pick the most efficient model for this prompt; fall back to the next
        // candidate if one is rate-limited, retired or erroring.
        let target = target_size(&content, mood);
        let candidates = self.pick_models(&key, target).await;
        let mut result: Result<String, String> = Err("no models available".to_string());
        for model in candidates {
            println!("Using model {model} (target ~{target}B)");
            result = self.ask_groq(&key, &model, &system, &prompt).await;
            match &result {
                Ok(_) => break,
                Err(e) if e == "401" => break,
                Err(e) => eprintln!("Model {model} failed, trying next: {e}"),
            }
        }

        let reply = match result {
            Ok(text) if !text.is_empty() => text,
            Ok(_) => "...".to_string(),
            Err(e) if e == "401" => {
                "my groq key got rejected. an admin needs to run /setup again.".to_string()
            }
            Err(e) => {
                eprintln!("Groq error: {e}");
                "servers got seized by capital. try again in a sec.".to_string()
            }
        };

        // Safety: never let the bot mass-ping.
        let reply = reply
            .replace("@everyone", "@\u{200b}everyone")
            .replace("@here", "@\u{200b}here");

        for (i, chunk) in split_message(&reply, 1900).iter().enumerate() {
            let result = if i == 0 {
                msg.reply(&ctx.http, chunk).await.map(|_| ())
            } else {
                msg.channel_id.say(&ctx.http, chunk).await.map(|_| ())
            };
            if let Err(e) = result {
                eprintln!("Failed to send message: {e}");
            }
        }
    }
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    let token = env::var("DISCORD_TOKEN").expect("Set DISCORD_TOKEN (the Bot token)");
    let keys_path = PathBuf::from(env::var("KEYS_FILE").unwrap_or_else(|_| "keys.json".to_string()));

    let intents = GatewayIntents::GUILDS
        | GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    let handler = Handler {
        http: reqwest::Client::new(),
        models: RwLock::new(None),
        keys: Arc::new(RwLock::new(load_keys(&keys_path))),
        keys_path,
    };

    let mut client = Client::builder(&token, intents)
        .event_handler(handler)
        .await
        .expect("Error creating Discord client");

    if let Err(e) = client.start().await {
        eprintln!("Client error: {e}");
    }
}
