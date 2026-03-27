use crate::engine::llama::LlamaEngine;
use crate::{config::RagConfig, engine::qdrant::QdrantEngine};

use log::LevelFilter;

mod api;
mod config;
mod engine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  let settings: RagConfig = crate::config::RagConfig::load(None)?;

  let log_level = if settings.log_debug {
    LevelFilter::Debug
  } else {
    LevelFilter::Info
  };
  let log_file = settings.log_file.to_owned();

  println!("using log file '{}'.", log_file);

  match simple_logging::log_to_file(&log_file, log_level) {
    Ok(_) => {}
    Err(e) => {
      eprintln!("could not create log file '{}': {}", log_file, e);
      simple_logging::log_to_stderr(log_level);
    }
  }

  log::info!("got settings: {:?}", settings);

  // then initialize llama
  log::debug!("initializing llama engine.");
  let llama = LlamaEngine::new()
    .with_llama_server(&settings.llama.chat_server)
    .with_embed_server(&settings.llama.embed_server)
    .with_rerank_server(&settings.llama.rerank_server);

  // then initialize qdrant
  let qdrant_url = format!("{}:{}", settings.qdrant.host, settings.qdrant.port);
  log::info!("initialize qdrant: {}", qdrant_url);

  let qdrant: QdrantEngine = QdrantEngine::new(qdrant_url);

  // then create the api object
  let api = api::RagApi::new()
    .with_listen_host(&settings.http.host)
    .with_listen_port(settings.http.port)
    .with_llama_engine(llama)
    .with_qdrant_engine(qdrant);

  // the do it
  match api.listen().await {
    Ok(v) => Ok(v),
    Err(e) => Err(e),
  }
}
