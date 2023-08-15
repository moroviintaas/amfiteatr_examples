use std::path::PathBuf;
use clap::{Parser, ValueEnum};
use log::LevelFilter;


#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct ExampleOptions{

    #[arg(short = 'l', long = "log_level", value_enum, default_value = "OFF")]
    pub log_level: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>

}