use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use log::LevelFilter;
use tch::Device;
use amfi::domain::RewardSource;
use clap::Parser;
use amfi::agent::{AgentGenT, AutomaticAgentRewarded};
use amfi::comm::SyncCommEnv;
use amfi_examples::classic::agent::HistorylessInfoSet;
use amfi_examples::classic::common::{AsymmetricRewardTableInt, SymmetricRewardTable};
use amfi_examples::classic::domain::{ClassicAction, ClassicGameDomainNumbered};
use amfi_examples::classic::policy::ClassicPureStrategy;
use amfi_examples::pairing::PairingState;


#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct ReplicatorOptions{

    #[arg(short = 'v', long = "log_level", value_enum, default_value = "OFF")]
    pub log_level: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

    #[arg(short = 's', long = "save")]
    pub save_file: Option<PathBuf>,

    #[arg(short = 'l', long = "load")]
    pub load_file: Option<PathBuf>,

    #[arg(short = 'e', long = "epochs", default_value = "10")]
    pub epochs: usize,
    
    #[arg(short = 'n', long = "rounds", default_value = "10")]
    pub number_of_rounds: usize
    

    //#[arg(short = 'r', long = "reward", default_value = "env")]
    //pub reward_source: RewardSource,

}

fn main(){

    let args = ReplicatorOptions::parse();
    let device = Device::Cpu;
    type Domain = ClassicGameDomainNumbered;

    let number_of_players = 32;
    let reward_table: AsymmetricRewardTableInt =
        SymmetricRewardTable::new(2, 1, 4, 0).into();
    let env_state_template = PairingState::new_even(number_of_players as u32, args.number_of_rounds, reward_table).unwrap();

    let mut comms = HashMap::<u32, SyncCommEnv<ClassicGameDomainNumbered>>::with_capacity(number_of_players);
    let mut agents = Vec::<Arc<Mutex<dyn AutomaticAgentRewarded<Domain>>>>::with_capacity(number_of_players);

    for i in 0..number_of_players/2{
        
        let comm_pair = SyncCommEnv::new_pair();
        let policy = ClassicPureStrategy::new(ClassicAction::Cooperate);
        let agent = AgentGenT::new( HistorylessInfoSet::new(i as u32, reward_table.clone()), comm_pair.1, policy);
        comms.insert(i as u32, comm_pair.0);

    }
}

