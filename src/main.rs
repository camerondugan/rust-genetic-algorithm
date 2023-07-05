use std::cmp::Ordering::Equal;
use rand::Rng;
use futures::future::join_all;

#[derive(Clone)]
pub struct Creature {
    brain: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>,
    in_size: usize,
    out_size: usize,
    mutate_speed: f32,
    fitness: f32,
}

impl Default for Creature {
    fn default() -> Creature {
        let mut rng = rand::thread_rng();
        let b = vec![(0..2).map(|_| {rng.gen_range(-1.0..1.0)}).collect()];
        let mut c = Creature {
            brain: b,
            weights: vec![],
            in_size: 2,
            out_size: 1,
            mutate_speed: 3.0,
            fitness: f32::MIN,
        };
        c.fix_weights();
        c
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut horde:Vec<Creature> = vec![];
    let death_rate = 0.8; //per epoch
    let horde_size = 100;
    let death_amount = (death_rate*(horde_size as f32)) as i32; //per epoch
   
    for _ in 0..horde_size {
        horde.push(Creature{in_size:2,out_size:1,..Default::default()});
    }
    
    futures::executor::block_on(async {
        let mut count = 0;
        while horde[0].fitness < 0.0 {

            let mut example_input = vec![];
            let mut example_output = vec![];

            example_input.push(vec![1.0,0.0]);
            example_input.push(vec![0.0,1.0]);
            example_input.push(vec![1.0,1.0]);
            example_input.push(vec![0.0,0.0]);

            example_output.push(vec![0.0]);
            example_output.push(vec![0.0]);
            example_output.push(vec![1.0]);
            example_output.push(vec![1.0]);

            join_all(horde
                     .iter_mut()
                     .map(|x| x.test_fitness(&example_input, &example_output)))
                .await;

            // https://www.reddit.com/r/rust/comments/29kia3/no_ord_for_f32/
            horde.sort_by(
                |a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Equal)
            );

            println!("brain size: {:?}", horde[0].brain_size());
            println!("brain mutate speed: {:?}", horde[0].mutate_speed);
            println!("brain: {:?}", horde[0].brain);
            

            println!("best for 0,0: {:?}", horde[0].feed(&vec![0.0,0.0]));
            println!("best for 0,1: {:?}", horde[0].feed(&vec![0.0,1.0]));
            println!("best for 1,0: {:?}", horde[0].feed(&vec![1.0,0.0]));
            println!("best for 1,1: {:?}", horde[0].feed(&vec![1.0,1.0]));

            for i in 0..10 {
                println!("{}th fitness: {:?}", i,horde[i].fitness);
            }


            let mut species_culled = vec![];

            let hlen = horde.len();
            let dedicated_survivor = (hlen as f32*0.6) as usize;
            for i in 0..death_amount as usize {
                horde.insert(i,horde[rng.gen_range(i..i+dedicated_survivor)].new_variant());
            }

            for _ in 0..death_amount {
                species_culled.push(horde.pop().unwrap());
            }
            // old method

           count += 1;
           println!("count: {}",count)
        }

    });
}

fn dist(v1:&Vec<f32>, v2:&Vec<f32>) -> f32 {
    let mut distance = 0.0;
    for i in 0..v1.len() {
        distance += (v1[i]-v2[i]).abs();
    }
    distance
}

// Works best (that I've tried) for xor
fn re_lu(x:f32) -> f32 {
    if x>=0.0 {x} else {0.0}
}

// For other applications
fn leaky_re_lu(x:f32) -> f32 {
    if x>=0.0 {x} else {0.01*x}
}

fn convolution(layer1:Vec<f32>, weights:&Vec<f32>, layer2:&Vec<f32>) -> Vec<f32> {
    let mut conv_output:Vec<f32> = Vec::with_capacity(layer2.len());
    
    let mut xi = 0;
    let l1_len = layer1.len();
    for x in layer2 {
        let mut sum:f32 = x.clone();
        for i in 0..l1_len {
            sum += weights[xi*l1_len+i]*layer1[i];
        }
        //conv_output.push(leaky_re_lu(sum));
        conv_output.push(re_lu(sum));
        xi+=1;
    }

    conv_output
}

impl Creature {
    pub fn feed(&self, input:&Vec<f32>) -> Vec<f32> {
        let mut current_layer = input.clone();

        for i in 0..self.brain.len() {
            current_layer = convolution(current_layer,&self.weights[i],&self.brain[i]);
        }

        convolution(current_layer,&self.weights[self.weights.len()-1],&vec![1.0;self.out_size])
    }

    pub fn new_variant(&self) -> Creature {
        let mut rng = rand::thread_rng();
        let mut variant = self.clone();

        let number = rng.gen_range(0..80);
        match number {
            1 => variant.add_random_node(),
            2 => variant.remove_random_node(),
            3 => variant.add_random_layer(),
            4 => variant.remove_random_layer(),
            5..=20 => variant.random_weights(),
            _ => variant.mutate_node(),
        }
        let change_more = rng.gen_range(0..1) != 0;
        if change_more { variant = variant.new_variant();}
        variant
    }

    pub fn brain_size(&self) -> usize {
        let mut size: usize = 0;
        for layer in &self.brain {
            size += layer.len();
        }
        size
    }

    pub async fn test_fitness(&mut self,input:&Vec<Vec<f32>>, expected:&Vec<Vec<f32>>) {
        let mut fitness = 0.0;
        for tup in input.iter().zip(expected) {
            let (inp, exp) = tup;
            fitness -= dist(&self.feed(inp),&exp)
        }
        self.fitness = fitness;
    }

    fn mutate_node(&mut self) {
        let mut rng = rand::thread_rng();
        let mut layer_i = rng.gen_range(0..self.brain.len());
        let start_layer = layer_i;
        while self.brain[layer_i].len() == 0 {
            layer_i+=1;
            layer_i%=self.brain.len();
            if layer_i == start_layer {return;}//infinite loop escape
        }
        let node_i = rng.gen_range(0..self.brain[layer_i].len());
        let modify_brain = rng.gen_range(0..=1) == 0;
        let mut nodes = &self.brain;
        if !modify_brain {
            nodes = &self.weights;
        }
        let mut node = nodes[layer_i][node_i];
        node += rng.gen_range(-self.mutate_speed..self.mutate_speed);
        let change_more = rng.gen_range(0..4) != 0;
        if change_more {
            self.mutate_node();
        }
        self.brain[layer_i][node_i] = node;
    }

    fn add_random_layer(&mut self) {
        let mut rng = rand::thread_rng();
        self.brain.push((0..rng.gen_range(2..4)).map(|_| {rng.gen_range(-1.0..1.0)}).collect());
        self.fix_weights();
    }

    fn add_random_node(&mut self) {
        let mut rng = rand::thread_rng();
        let new_node = rng.gen_range(-1.0..1.0);
        let layer_i = rng.gen_range(0..self.brain.len());
        if self.brain[layer_i].len() == 0 {
            self.brain[layer_i].push(new_node);
        } else {
            let node_i = rng.gen_range(0..self.brain[layer_i].len());
            self.brain[layer_i].insert(node_i, new_node);

        }
        self.fix_weights();
    }

    fn remove_random_node(&mut self) {
        if self.brain_size() <= 3 {return;}
        let mut rng = rand::thread_rng();
        if self.brain.len() <= 0 {return};
        let layer_i = rng.gen_range(0..self.brain.len());
        if self.brain[layer_i].len() <= 2 {return};
        let node_i = rng.gen_range(0..self.brain[layer_i].len());
        self.brain[layer_i].remove(node_i);
        self.fix_weights();
    }

    fn remove_random_layer(&mut self) {
        if self.brain_size() <= 3 {return;}
        let mut rng = rand::thread_rng();
        if self.brain.len() <= 1 {return};
        let layer_i = rng.gen_range(0..self.brain.len());
        self.brain.remove(layer_i);
        self.weights.remove(layer_i);
        self.fix_weights();
    }


    fn random_weights(&mut self) {
        self.weights.clear();
        self.fix_weights();
    }

    fn fix_weights(&mut self) {
        let mut rng = rand::thread_rng();
        //let mut w: Vec<Vec<f32>> = vec![];
        let mut last_len: usize = self.in_size;
        if self.weights.len() < self.brain.len() { //missing layers
            for _ in self.weights.len()..self.brain.len() {
                self.weights.push(vec![]); //add empty layers
            }
        }
        for i in 0..self.brain.len() { // missing nodes
                if self.brain[i].len() == 0 {continue;};
                let mut weights : Vec<f32> = (self.weights[i].len()..last_len*self.brain[i].len()).map(|_| 
                rng.gen_range(-self.mutate_speed..self.mutate_speed)).collect();
                self.weights[i].append(&mut weights);
                last_len = self.brain[i].len();
        }

        //rng.gen_range(-self.mutate_speed..self.mutate_speed)).collect());
    }
}
