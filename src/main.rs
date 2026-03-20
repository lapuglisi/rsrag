use crate::config::RagConfig;
use crate::engine::llama::LlamaEngine;

mod api;
mod config;
mod engine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  let settings: RagConfig = crate::config::RagConfig::load(None)?;

  println!("got settings: {:?}", settings);

  // then initialize llama
  let llama = LlamaEngine::new()
    .with_llama_server(&settings.llama.chat_server)
    .with_embed_server(&settings.llama.embed_server)
    .with_rerank_server(&settings.llama.rerank_server);

  // then initialize qdrant

  // then create the api object
  let api = api::RagApi::new()
    .with_listen_host(&settings.http.host)
    .with_listen_port(settings.http.port)
    .with_llama_engine(llama);

  // the do it
  api.listen().await;

  Ok(())
}
