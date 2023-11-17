use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use log::LevelFilter;
use tch::Device;
use amfi::domain::RewardSource;
use clap::Parser;
use amfi::agent::{AgentGenT, AutomaticAgentBothPayoffs, AutomaticAgentRewarded, ScoringInformationSet, TracingAutomaticAgent};
use amfi::comm::SyncCommEnv;
use amfi::env::generic::HashMapEnv;
use amfi::env::RoundRobinUniversalEnvironment;
use amfi_examples::classic::agent::{BoxedClassicInfoSet, HistorylessInfoSet};
use amfi_examples::classic::common::{AsymmetricRewardTableInt, SymmetricRewardTable};
use amfi_examples::classic::domain::{ClassicAction, ClassicGameDomainNumbered, IntReward};
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
    let mut agents = Vec::<Arc<Mutex<Box<dyn AutomaticAgentBothPayoffs <Domain, InternalReward= IntReward> + Send>>>>::with_capacity(number_of_players);

    for i in 0..number_of_players/2{
        
        let comm_pair = SyncCommEnv::new_pair();
        let policy = ClassicPureStrategy::new(ClassicAction::Cooperate);
        let agent = AgentGenT::new( HistorylessInfoSet::new(i as u32, reward_table.clone()), comm_pair.1, policy);
        comms.insert(i as u32, comm_pair.0);
        agents.push(Arc::new(Mutex::new(Box::new(agent))));
    }
    for i in number_of_players/2..number_of_players{

        let comm_pair = SyncCommEnv::new_pair();
        let policy = ClassicPureStrategy::new(ClassicAction::Defect);
        let agent = AgentGenT::new( HistorylessInfoSet::new(i as u32, reward_table.clone()), comm_pair.1, policy);
        comms.insert(i as u32, comm_pair.0);
        agents.push(Arc::new(Mutex::new(Box::new(agent))));
    }

    let mut environment = HashMapEnv::new(env_state_template.clone(), comms);

    thread::scope(|s|{
        s.spawn(||{
            environment.run_round_robin_uni_rewards().unwrap();
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
    //println!("{:}", a.)

}

