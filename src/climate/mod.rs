//! Phase 4: Climate simulation.
//!
//! Produces monthly temperature + precipitation fields and annual summaries.

mod config;
mod util;
mod temperature;
mod wind;
mod coast;
mod moisture;

pub use config::ClimateConfig;
pub use coast::compute_coast_distance_km;
pub use moisture::{ClimatePrecompute, MonthlyClimate, precompute_lat_lon, compute_month};

