use config::{Config, ConfigError, Environment, File};
use serde::Deserialize;
use std::env;

use crate::engine::llama::LlamaEngine;

#[derive(Deserialize, Clone, Debug, Default)]
pub struct RagConfig {
  pub http: HttpConfig,
  pub llama: LlamaConfig,
  pub qdrant: QdrantConfig,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct HttpConfig {
  pub host: String,
  pub port: u16,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct LlamaConfig {
  pub chat_server: String,
  pub embed_server: String,
  pub rerank_server: String,
}

#[derive(Deserialize, Clone, Debug, Default)]
pub struct QdrantConfig {
  pub host: String,
  pub port: u16,
  pub query_limit: u64,
}

// Add defaults
impl RagConfig {
  pub fn default() -> Self {
    Self {
      http: HttpConfig {
        host: "127.0.0.1".into(),
        port: 9091,
      },
      llama: LlamaConfig {
        chat_server: "127.0.0.1".into(),
        embed_server: "127.0.0.1".into(),
        rerank_server: "127.0.0.1".into(),
      },
      qdrant: QdrantConfig {
        host: "127.0.0.1".into(),
        port: 6334,
        query_limit: 0,
      },
    }
  }

  pub fn load(cfg: Option<String>) -> Result<Self, ConfigError> {
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
      .set_default("llama.embed_server", "buscosexo")?
      .add_source(File::with_name(&config_path))
      .add_source(Environment::with_prefix("RSRAG").separator("_"))
      .build()?;

    Ok(config.try_deserialize::<RagConfig>().unwrap())
  }
}
