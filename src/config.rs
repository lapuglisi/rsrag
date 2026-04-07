use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;
use std::env;

const RAGAPI_DEFAULT_HOST: &str = "127.0.0.1";
const RAGAPI_DEFAULT_PORT: u16 = 9091;
const RAGAPI_DEFAULT_RAG_STRATEGY: &str = "main-dense";

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct RagConfig {
  pub log_file: Option<String>,
  pub log_debug: bool,
  pub rag_strategy: String,
  pub http: HttpConfig,
  pub llama: LlamaConfig,
  pub qdrant: QdrantConfig,
}

impl Default for RagConfig {
  fn default() -> Self {
    Self {
      log_file: None,
      log_debug: false,
      rag_strategy: String::from(RAGAPI_DEFAULT_RAG_STRATEGY),
      http: HttpConfig::default(),
      llama: LlamaConfig::default(),
      qdrant: QdrantConfig::default(),
    }
  }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct HttpConfig {
  pub host: String,
  pub port: u16,
}

impl Default for HttpConfig {
  fn default() -> Self {
    Self {
      host: RAGAPI_DEFAULT_HOST.to_string(),
      port: RAGAPI_DEFAULT_PORT,
    }
  }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct LlamaConfig {
  pub chat_server: String,
  pub embed_server: String,
  pub rerank_server: String,
}

impl Default for LlamaConfig {
  fn default() -> Self {
    Self {
      chat_server: "127.0.0.1".to_string(),
      embed_server: "127.0.0.1".to_string(),
      rerank_server: "127.0.0.1".to_string(),
    }
  }
}

#[derive(Deserialize, Clone, Debug)]
#[serde(default)]
pub struct QdrantConfig {
  pub host: String,
  pub port: u16,
  pub query_limit: u64,
  pub collection: String,
}

impl Default for QdrantConfig {
  fn default() -> Self {
    Self {
      host: "127.0.0.1".to_string(),
      port: 6334,
      query_limit: 0,
      collection: String::new(),
    }
  }
}

// Add defaults
impl RagConfig {
  pub fn load(cfg: Option<String>) -> Result<Self, ConfigError> {
    // TODO: use unwrap_or[_else] when defining path/home.
    let path: String = env::var("XDG_CONFIG_HOME").unwrap_or_default();
    let home: String = env::var("HOME").unwrap_or_default();
    let config_path: String;

    if cfg.is_some() {
      config_path = cfg.unwrap();
    } else if !path.is_empty() {
      config_path = format!("{}/rsrag/rsrag.toml", path);
    } else if !home.is_empty() {
      config_path = format!("{}/.config/rsrag/rsrag.toml", home);
    } else {
      config_path = "config/rsrag.toml".to_string();
    }

    println!("[rsrag] using config file: {}", config_path);

    let config = Config::builder()
      .add_source(File::with_name(&config_path))
      .add_source(Environment::with_prefix("RSRAG").separator("_"))
      .build()?;

    Ok(config.try_deserialize::<RagConfig>().unwrap())
  }
}
