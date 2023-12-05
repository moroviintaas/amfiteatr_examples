mod options;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use log::{debug, info, LevelFilter};
use tch::{Device, nn, Tensor};
use clap::Parser;
use enum_map::Enum;
use tch::nn::{Adam, VarStore};
use amfi::agent::*;
use amfi::comm::{AgentMpscPort, EnvMpscPort, SyncCommAgent};
use amfi::env::generic::BasicEnvironment;
use amfi::env::{AutoEnvironmentWithScores, ReseedEnvironment};
use amfi_classic::policy::{ClassicMixedStrategy, ClassicPureStrategy};
use amfi::agent::AgentWithId;
use amfi::agent::InternalRewardedAgent;
use amfi::agent::EnvRewardedAgent;
use amfi::agent::TracingAgent;
use amfi::domain::DomainParameters;
use amfi::error::AmfiError;
use amfi_classic::domain::{AgentNum, ClassicAction, ClassicGameDomainNumbered, IntReward};
use amfi_classic::env::PairingState;
use amfi_classic::{AsymmetricRewardTableInt, SymmetricRewardTable};
use amfi_classic::agent::{HistorylessInfoSet, OwnHistoryInfoSet, OwnHistoryTensorRepr, VerboseReward};
use amfi_classic::domain::ClassicAction::Defect;
use amfi_rl::actor_critic::ActorCriticPolicy;
use amfi_rl::agent::RlModelAgent;
use amfi_rl::{LearningNetworkPolicy, TrainConfig};
use amfi_rl::tensor_repr::WayToTensor;
use amfi_rl::torch_net::{A2CNet, NeuralNetTemplate, TensorA2C};
use crate::options::ReplicatorOptions;


pub fn avg(entries: &[f64]) -> Option<f64>{
    if entries.is_empty(){
        None
    } else {
        let sum = entries.iter().sum::<f64>();
        Some(sum / entries.len() as f64)
    }
}

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
        .level_for("amfi_examples", options.log_level)
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
type MixedPolicy = ClassicMixedStrategy<AgentNum, OwnHistoryInfoSet<AgentNum>>;
type PurePolicy = ClassicPureStrategy<AgentNum, OwnHistoryInfoSet<AgentNum>>;
type AgentComm = AgentMpscPort<D>;

pub enum Group{
    Mixes,
    Hawks,
    Doves,
    Learning
}

struct Model{
    pub environment: BasicEnvironment<D, S, EnvMpscPort<D>>,
    //agents: Arc<Mutex<dyn MultiEpisodeAgent<D, (), InfoSetType=()>>>,
    pub mixed_agents: Vec<Arc<Mutex<AgentGen<D, MixedPolicy, AgentComm>>>>,
    pub hawk_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>>,
    pub dove_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>>,
    pub learning_agents: Vec<Arc<Mutex<AgentGenT<D, Pol, AgentComm>>>>,
    
    pub averages_mixed: Vec<f64>,
    pub averages_hawk: Vec<f64>,
    pub averages_dove: Vec<f64>,
    pub averages_learning: Vec<f64>,
    pub averages_all: Vec<f64>,

    scores_mixed: Vec<f64>,
    scores_hawk: Vec<f64>,
    scores_dove: Vec<f64>,
    scores_learning: Vec<f64>,
    scores_all: Vec<f64>,
}

impl Model{

    pub fn new(environment: BasicEnvironment<D, S, EnvMpscPort<D>>) -> Self{
        Self{
            environment, mixed_agents: Vec::new(), hawk_agents: Vec::new(), dove_agents: Vec::new(),
            learning_agents: Vec::new(), averages_mixed: Vec::new(),
            averages_hawk: Vec::new(),
            averages_dove: Vec::new(),
            averages_learning: Vec::new(),
            averages_all: Vec::new(),
            scores_mixed: vec![],
            scores_hawk: vec![],
            scores_dove: vec![],
            scores_learning: vec![],
            scores_all: vec![],
        }
    }

    pub fn new_with_agents(environment: BasicEnvironment<D, S, EnvMpscPort<D>>,
                           learning_agents: Vec<Arc<Mutex<AgentGenT<D, Pol, AgentComm>>>>,
        mixed_agents: Vec<Arc<Mutex<AgentGen<D, MixedPolicy, AgentComm>>>>,
        hawk_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>>,
        dove_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>>,
        ) -> Self{
        Self{
            environment, learning_agents,
            mixed_agents, hawk_agents, dove_agents,
            averages_mixed: vec![],
            averages_hawk: vec![],
            averages_dove: vec![],
            averages_learning: vec![],
            averages_all: vec![],
            scores_mixed: vec![],
            scores_hawk: vec![],
            scores_dove: vec![],
            scores_learning: vec![],


            scores_all: vec![],
        }

    }
    pub fn clear_averages(&mut self){
        self.averages_dove.clear();
        self.averages_hawk.clear();
        self.averages_learning.clear();
        self.averages_all.clear();
        self.averages_mixed.clear();
    }

    pub fn clear_trajectories(&mut self){

        for agent in &self.learning_agents{
            let mut guard = agent.lock().unwrap();
            guard.reset_trajectory()
        }
        /*
        for agent in self.hawk_agents{
            let mut guard = agent.lock().unwrap();
            guard.reset_trajectory()
        }
        for agent in self.dove_agents{
            let mut guard = agent.lock().unwrap();
            guard.reset_trajectory()
        }
        for agent in self.mixed_agents{
            let mut guard = agent.lock().unwrap();
            guard.reset_trajectory()
        }

         */
    }

    pub fn remember_average_group_scores(&mut self){
        self.clear_episode_scores();

        for agent in &self.learning_agents{
            let guard = agent.lock().unwrap();
            let score = guard.current_universal_score() as f64;
            self.scores_all.push(score);
            self.scores_learning.push(score);
        }
        for agent in &self.mixed_agents{
            let guard = agent.lock().unwrap();
            let score = guard.current_universal_score() as f64;
            self.scores_all.push(score);
            self.scores_mixed.push(score);
        }
        for agent in &self.dove_agents{
            let guard = agent.lock().unwrap();
            let score = guard.current_universal_score() as f64;
            self.scores_all.push(score);
            self.scores_dove.push(score);
        }
        for agent in &self.hawk_agents{
            let guard = agent.lock().unwrap();
            let score = guard.current_universal_score() as f64;
            self.scores_all.push(score);
            self.scores_hawk.push(score);
        }

        if let Some(average) = avg(&self.scores_dove[..]){
            self.averages_dove.push(average)
        }
        if let Some(average) = avg(&self.scores_hawk[..]){
            self.averages_hawk.push(average)
        }
        if let Some(average) = avg(&self.scores_mixed[..]){
            self.averages_mixed.push(average)
        }
        if let Some(average) = avg(&self.scores_learning[..]){
            self.averages_learning.push(average)
        }
        if let Some(average) = avg(&self.scores_all[..]){
            self.averages_all.push(average)
        }

    }

    pub fn clear_episode_scores(&mut self){
        self.scores_mixed.clear();
        self.scores_all.clear();
        self.scores_dove.clear();
        self.scores_hawk.clear();
        self.scores_learning.clear();
    }

    pub fn run_episode(&mut self) -> Result<(), AmfiError<D>>{

        thread::scope(|s|{
            s.spawn(||{
                self.environment.reseed(());
                self.environment.run_with_scores().unwrap();
            });
            for a in  &self.dove_agents{
                let agent = a.clone();
                s.spawn(move ||{
                    let mut guard = agent.lock().unwrap();
                    guard.run_episode_rewarded(()).unwrap();

                });
            };

            for a in  &self.hawk_agents{
                let agent = a.clone();
                s.spawn(move ||{
                    let mut guard = agent.lock().unwrap();
                    guard.run_episode_rewarded(()).unwrap();

                });
            };

            for a in  &self.mixed_agents{
                let agent = a.clone();
                s.spawn(move ||{
                    let mut guard = agent.lock().unwrap();
                    guard.run_episode_rewarded(()).unwrap();

                });
            };

            for a in  &self.learning_agents{
                let agent = a.clone();
                s.spawn(move ||{
                    let mut guard = agent.lock().unwrap();
                    guard.run_episode_rewarded(()).unwrap();

                });
            };



        });

        Ok(())
    }

    pub fn update_policies(&mut self) -> Result<(), AmfiError<D>>{
        for a in &self.learning_agents{
            let mut agent = a.lock().unwrap();
            let trajectories = agent.take_episodes();
            agent.policy_mut().train_on_trajectories_env_reward(&trajectories[..])?;
        }
        Ok(())
    }
}


fn main() -> Result<(), AmfiError<D>>{
    debug!("Starting");

    let args = ReplicatorOptions::parse();
    setup_logger(&args).unwrap();
    let device = Device::Cpu;
    //let device = Device::Cpu;

    let number_of_players = 32;
    let reward_table: AsymmetricRewardTableInt =
        SymmetricRewardTable::new(2, 1, 4, 0).into();
    //let env_state_template = PairingState::new_even(number_of_players, args.number_of_rounds, reward_table).unwrap();
    let tensor_repr = OwnHistoryTensorRepr::new(args.number_of_rounds);
    let input_size = tensor_repr.desired_shape().iter().product();
    //let mut comms = HashMap::<u32, SyncCommEnv<ClassicGameDomainNumbered>>::with_capacity(number_of_players);

    let net_template = NeuralNetTemplate::new(|path|{
        let seq = nn::seq()
            .add(nn::linear(path / "input", input_size, 512, Default::default()))
            //.add(nn::linear(path / "h1", 256, 256, Default::default()))
            .add(nn::linear(path / "hidden1", 512, 512, Default::default()))
            .add_fn(|xs|xs.relu());
            //.add(nn::linear(path / "h2", 512, 512, Default::default()));
        let actor = nn::linear(path / "al", 512, 2, Default::default());
        let critic =  nn::linear(path / "ac", 512, 1, Default::default());
        {move |input: &Tensor|{
            let xs = input.to_device(device).apply(&seq);
            TensorA2C{critic: xs.apply(&critic), actor: xs.apply(&actor)}
        }}
    });

    let mut env_adapter = EnvMpscPort::new();

    let mut learning_agents: Vec<Arc<Mutex<AgentGenT<D, Pol, AgentComm>>>> = Vec::new();
    let mut mixed_agents: Vec<Arc<Mutex<AgentGen<D, MixedPolicy, AgentComm>>>> = Vec::new();
    let mut hawk_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>> = Vec::new();
    let mut dove_agents: Vec<Arc<Mutex<AgentGen<D, PurePolicy, AgentComm>>>> = Vec::new();


    let offset_learning = 0 as AgentNum;
    let offset_mixed = args.number_of_learning as AgentNum;
    let offset_hawk = args.number_of_mixes as AgentNum + offset_mixed;
    let offset_dove = args.number_of_hawks as AgentNum + offset_hawk;
    let total_number_of_players = offset_dove as usize + args.number_of_doves;

    for i in 0..offset_mixed{
        let comm = env_adapter.register_agent(i)?;
        let state = OwnHistoryInfoSet::new(i, reward_table);
        let net = A2CNet::new(VarStore::new(device), net_template.get_net_closure());
        let opt = net.build_optimizer(Adam::default(), 1e-4).unwrap();
        let policy = ActorCriticPolicy::new(net, opt, tensor_repr, TrainConfig {gamma: 0.99});
        let agent = AgentGenT::new(state, comm, policy);
        learning_agents.push(Arc::new(Mutex::new(agent)));

    }
    debug!("Created learning agent vector");

    for i in offset_mixed..offset_hawk{
        let comm = env_adapter.register_agent(i)?;
        let state = OwnHistoryInfoSet::new(i, reward_table);

        let policy = MixedPolicy::new(args.mix_probability_of_hawk);
        let agent = AgentGen::new(state, comm, policy);
        mixed_agents.push(Arc::new(Mutex::new(agent)));
    }

    for i in offset_hawk..offset_dove{
        let comm = env_adapter.register_agent(i)?;
        let state = OwnHistoryInfoSet::new(i, reward_table);

        let policy = PurePolicy::new(ClassicAction::Defect);
        let agent = AgentGen::new(state, comm, policy);
        hawk_agents.push(Arc::new(Mutex::new(agent)));

    }
    for i in offset_dove..total_number_of_players as AgentNum{
        let comm = env_adapter.register_agent(i)?;
        let state = OwnHistoryInfoSet::new(i, reward_table);

        let policy = PurePolicy::new(ClassicAction::Cooperate);
        let agent = AgentGen::new(state, comm, policy);
        dove_agents.push(Arc::new(Mutex::new(agent)));

    }
    let env_state = PairingState::new_even(total_number_of_players,
                                           args.number_of_rounds, reward_table)?;
    let environment = BasicEnvironment::new(env_state, env_adapter);


    let mut model = Model::new_with_agents(environment, learning_agents, mixed_agents,
                                           hawk_agents, dove_agents);

    // inital test

    info!("Starting initial evaluation");
    for i in 0..100{
        model.run_episode()?;
        model.remember_average_group_scores();

    }
    if let Some(average) = avg(&model.averages_learning){
        info!("Average learning agent score in {} rounds: {}", args.number_of_rounds, average );
    }
    if let Some(average) = avg(&model.averages_dove){
        info!("Average dove agent score in {} rounds: {}", args.number_of_rounds, average );
    }
    if let Some(average) = avg(&model.averages_hawk){
        info!("Average hawk agent score in {} rounds: {}", args.number_of_rounds, average );
    }
    if let Some(average) = avg(&model.averages_mixed){
        info!("Average mixed({}) agent score in {} rounds: {}", args.mix_probability_of_hawk , args.number_of_rounds, average );
    }
    if let Some(average) = avg(&model.averages_all){
        info!("Average any agent score in {} rounds: {}", args.number_of_rounds, average );
    }

    /*
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

        let agnt_comm = env_adapter.register_agent(i as u32).unwrap();
        let agent = AgentGenT::new( HistorylessInfoSet::new(i as u32, reward_table.clone()), agnt_comm, policy);
        agents.push(Arc::new(Mutex::new(agent)));
    }


     */
    Ok(())

}

