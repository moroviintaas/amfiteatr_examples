use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use log::LevelFilter;
use tch::Device;
use amfi::domain::RewardSource;
use clap::Parser;
use amfi::agent::{AgentGenT, AutomaticAgentBothPayoffs, AutomaticAgentRewarded, ScoringInformationSet, TracingAutomaticAgent};
use amfi::comm::{EnvMpscPort, SyncCommEnv};
use amfi::env::generic::{BasicEnvironment, HashMapEnv};
use amfi::env::{AutoEnvironment, AutoEnvironmentWithScores, RoundRobinUniversalEnvironment};
use amfi_classic::policy::ClassicPureStrategy;
use amfi::agent::AgentWithId;
use amfi::agent::InternalRewardedAgent;
use amfi::agent::EnvRewardedAgent;
use amfi::agent::TracingAgent;
use amfi_classic::domain::{ClassicAction, ClassicGameDomainNumbered};
use amfi_classic::env::PairingState;
use amfi_classic::{AsymmetricRewardTableInt, SymmetricRewardTable};
use amfi_classic::agent::HistorylessInfoSet;


#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct ReplicatorOptions{

    #[arg(short = 'v', long = "log_level", value_enum, default_value = "debug")]
    pub log_level: LevelFilter,

    #[arg(short = 'a', long = "log_level_amfi", value_enum, default_value = "OFF")]
    pub log_level_amfi: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

    #[arg(short = 's', long = "save")]
    pub save_file: Option<PathBuf>,

    #[arg(short = 'l', long = "load")]
    pub load_file: Option<PathBuf>,

    #[arg(short = 'e', long = "epochs", default_value = "10")]
    pub epochs: usize,
    
    #[arg(short = 'n', long = "rounds", default_value = "32")]
    pub number_of_rounds: usize
    

    //#[arg(short = 'r', long = "reward", default_value = "env")]
    //pub reward_source: RewardSource,

}

pub fn setup_logger(options: &ReplicatorOptions) -> Result<(), fern::InitError> {
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(options.log_level)
        //.level_for("amfi_examples", options.log_level)
        //.level_for("pairing", options.log_level)
        //.level_for("classic", options.log_level);
        .level_for("amfi", options.log_level_amfi);

        match &options.log_file{
            None => dispatch.chain(std::io::stdout()),
            Some(f) => dispatch.chain(fern::log_file(f)?)
        }

        .apply()?;
    Ok(())
}

fn main(){

    let args = ReplicatorOptions::parse();
    let device = Device::Cpu;
    type Domain = ClassicGameDomainNumbered;

    let number_of_players = 32;
    let reward_table: AsymmetricRewardTableInt =
        SymmetricRewardTable::new(2, 1, 4, 0).into();
    let env_state_template = PairingState::new_even(number_of_players, args.number_of_rounds, reward_table).unwrap();

    let mut comms = HashMap::<u32, SyncCommEnv<ClassicGameDomainNumbered>>::with_capacity(number_of_players);

    let mut agents = Vec::with_capacity(number_of_players);

    let mut env_adapter = EnvMpscPort::new();


    for i in 0..number_of_players/2{
        let policy = ClassicPureStrategy::new(ClassicAction::Cooperate);
        //let comm_pair = SyncCommEnv::new_pair();
        //let agent = AgentGenT::new( HistorylessInfoSet::new(i as u32, reward_table.clone()), comm_pair.1, policy);
        //comms.insert(i as u32, comm_pair.0);

        let agnt_comm = env_adapter.register_agent(i as u32).unwrap();
        let agent = AgentGenT::new( HistorylessInfoSet::new(i as u32, reward_table.clone()), agnt_comm, policy);
        agents.push(Arc::new(Mutex::new(agent)));
    }
    for i in number_of_players/2..number_of_players{
        let policy = ClassicPureStrategy::new(ClassicAction::Defect);
        //let comm_pair = SyncCommEnv::new_pair();

        //let agent = AgentGenT::new( HistorylessInfoSet::new(i as u32, reward_table.clone()), comm_pair.1, policy);
        //comms.insert(i as u32, comm_pair.0);
        let agnt_comm = env_adapter.register_agent(i as u32).unwrap();
        let agent = AgentGenT::new( HistorylessInfoSet::new(i as u32, reward_table.clone()), agnt_comm, policy);
        agents.push(Arc::new(Mutex::new(agent)));
    }

    //let mut environment = HashMapEnv::new(env_state_template.clone(), comms);
    let mut environment = BasicEnvironment::new(env_state_template.clone(), env_adapter);
    thread::scope(|s|{
        s.spawn(||{
            environment.run_with_scores().unwrap();
        });

        for mut a in &agents{
            let mut agent = a.clone();
            s.spawn(move ||{

                agent.lock().unwrap().run_rewarded().unwrap();
            });
        }

    });

    for agent in &agents{
        let agent_lock = agent.lock().unwrap();

        let id = agent_lock.id();
        let score_uni = agent_lock.current_universal_score();
        let score_infoset = agent_lock.current_subjective_score();

        println!("Score of agent {id:}, universal: {score_uni:}, internal: {score_infoset:?}");
    }

    let a = agents[0].lock().unwrap();
    for tl in a.game_trajectory().list(){
        println!("{:}", tl);
    }
    

}

