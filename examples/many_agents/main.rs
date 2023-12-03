mod options;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use log::LevelFilter;
use tch::Device;
use clap::Parser;
use amfi::agent::*;
use amfi::comm::{EnvMpscPort};
use amfi::env::generic::BasicEnvironment;
use amfi::env::AutoEnvironmentWithScores;
use amfi_classic::policy::ClassicPureStrategy;
use amfi::agent::AgentWithId;
use amfi::agent::InternalRewardedAgent;
use amfi::agent::EnvRewardedAgent;
use amfi::agent::TracingAgent;
use amfi::domain::DomainParameters;
use amfi_classic::domain::{AgentNum, ClassicAction, ClassicGameDomainNumbered, IntReward};
use amfi_classic::env::PairingState;
use amfi_classic::{AsymmetricRewardTableInt, SymmetricRewardTable};
use amfi_classic::agent::{HistorylessInfoSet, OwnHistoryInfoSet, OwnHistoryTensorRepr, VerboseReward};
use amfi_rl::actor_critic::ActorCriticPolicy;
use amfi_rl::agent::RlModelAgent;
use crate::options::ReplicatorOptions;


pub fn setup_logger(options: &ReplicatorOptions) -> Result<(), fern::InitError> {
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%H:%M:%S]"),
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
type D = ClassicGameDomainNumbered;
type S = PairingState<<D as DomainParameters>::AgentId>;
type Pol = ActorCriticPolicy<D, OwnHistoryInfoSet<<D as DomainParameters>::AgentId>, OwnHistoryTensorRepr>;

struct Model{
    environment: BasicEnvironment<D, S, EnvMpscPort<D>>,
    //agents: Arc<Mutex<dyn MultiEpisodeAgent<D, (), InfoSetType=()>>>,
    basic_agents: Arc<Mutex<dyn MultiEpisodeAgent<D, (), InfoSetType=OwnHistoryInfoSet<AgentNum>>>>,
    learning_agents: Arc<Mutex<dyn RlModelAgent<D, (), OwnHistoryInfoSet<AgentNum>,
        Policy=Pol,>>>
}


fn main(){

    let args = ReplicatorOptions::parse();
    //let device = Device::Cpu;
    type Domain = ClassicGameDomainNumbered;

    let number_of_players = 32;
    let reward_table: AsymmetricRewardTableInt =
        SymmetricRewardTable::new(2, 1, 4, 0).into();
    let env_state_template = PairingState::new_even(number_of_players, args.number_of_rounds, reward_table).unwrap();

    //let mut comms = HashMap::<u32, SyncCommEnv<ClassicGameDomainNumbered>>::with_capacity(number_of_players);

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

        for a in &agents{
            let agent = a.clone();
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

