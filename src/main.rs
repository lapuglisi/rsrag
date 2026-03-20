use axum::{Json, Router, http::StatusCode, routing::post};
use config::Config;
use serde::{Deserialize, Serialize};

mod engine;

#[tokio::main]
async fn main() {
  let c = Config::builder()
    .add_source(config::File::with_name("busco"))
    .build()
    .expect("could not load config.");

  println!(
    "settings: {:?}",
    c.try_deserialize::<std::collections::HashMap<String, String>>()
      .unwrap()
  );

  let address = "127.0.0.1:9999";
  let app: Router = Router::new().route("/api/fodase", post(fodase));

  let listener = tokio::net::TcpListener::bind(address)
    .await
    .expect(format!("Could not bind to address {}", address).as_str());

  println!("[rsrag] listening on '{}'...", address);

  engine::hello();

  let _ = axum::serve(listener, app).await;
}

// post requests
async fn fodase(Json(payload): Json<Fodase>) -> (StatusCode, Json<Busco>) {
  let busco = Busco {
    answer: String::from("Amigos felizes"),
  };

  (StatusCode::OK, Json(busco))
}

#[derive(Deserialize)]
struct Fodase {
  model: Option<String>,
  prompt: String,
}

#[derive(Serialize)]
struct Busco {
  answer: String,
}
