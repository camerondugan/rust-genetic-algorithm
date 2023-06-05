use std::cmp::Ordering::Equal;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    let mut horde:Vec<Creature> = vec![];
    let death_rate = 700; //per epoch
   
    for _ in 0..1000 {
        horde.push(Creature{in_size:2,out_size:1,..Default::default()});
    }
    
    let mut count = 0;
    while horde[0].fitness < -0.2 {

        for mut creature in &mut horde {
            let mut fitness = 0.0;
            let mut out = creature.feed(vec![0.0,0.0]);
            fitness -= dist(out,vec![0.0]);
            out = creature.feed(vec![1.0,0.0]);
            fitness -= dist(out,vec![1.0]);
            out = creature.feed(vec![0.0,1.0]);
            fitness -= dist(out,vec![1.0]);
            out = creature.feed(vec![1.0,1.0]);
            fitness -= dist(out,vec![1.0]);
            creature.fitness = fitness;
        }

        // https://www.reddit.com/r/rust/comments/29kia3/no_ord_for_f32/
        horde.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(Equal));
        
        for i in 0..10 {
            println!("fitness: {:?}", horde[i].fitness);
        }
        println!("brain length: {:?}", horde[0].brain.len());
        println!("best for 0,0: {:?}", horde[0].feed(vec![0.0,0.0]));
        println!("best for 0,1: {:?}", horde[0].feed(vec![0.0,1.0]));
        println!("best for 1,0: {:?}", horde[0].feed(vec![1.0,0.0]));
        println!("best for 1,1: {:?}", horde[0].feed(vec![1.0,1.0]));

        let mut species_culled = vec![];

        for _ in 0..death_rate {
            species_culled.push(horde.pop().unwrap());
        }
        let hlen = horde.len();
        for _ in 0..death_rate {
            horde.push(horde[rng.gen_range(0..hlen/3)].new_variant());
        }

       // species_culled.sort_by_key(|s| s.species);

       // for creature in species_culled {
       //     let mut found = false;
       //     for i in 0..hlen {
       //         if horde[i].species == creature.species {
       //             horde.push(horde[i].new_variant());
       //             found = true;
       //             break;
       //         }
       //     }
       //     if !found {
       //         horde.push(creature.new_variant());
       //     }
       // }
        count += 1;
        println!("count: {}",count)
    }
    //});
    //handle.join().unwrap();
}

fn dist(v1:Vec<f32>, v2:Vec<f32>) -> f32 {
    let mut distance = 0.0;
    for i in 0..v1.len() {
        distance += (v1[i]-v2[i]).abs();
    }
    distance
}

fn convolution(layer1:Vec<f32>, weights:&Vec<f32>, layer2:&Vec<f32>) -> Vec<f32> {
    //println!("Convolution: {:?} {:?}",layer1,layer2);
    let mut conv_output:Vec<f32> = Vec::with_capacity(layer2.len());
    
    let mut xi = 0;
    let l1_len = layer1.len();
    for x in layer2 {
        let mut sum:f32 = x.clone();
        for i in 0..l1_len {
            sum += weights[xi*l1_len+i]*layer1[i];
        }
        conv_output.push(sum);
        //println!("{:?}", conv_output);
        xi+=1;
    }

    conv_output
}

#[derive(Clone)]
pub struct Creature {
    brain: Vec<Vec<f32>>,
    weights: Vec<Vec<f32>>,
    in_size: usize,
    out_size: usize,
    mutate_speed: f32,
    fitness: f32,
    species: i8,
}

impl Default for Creature {
    fn default() -> Creature {
        let mut rng = rand::thread_rng();
        let b = vec![(0..2).map(|_| {rng.gen_range(-1.0..1.0)}).collect()];
        let mut c = Creature {
            brain: b,
            weights: vec![],//temp until overridden
            in_size: 2,
            out_size: 1,
            mutate_speed: 4.0,
            fitness: -9999.9999,
            species: rng.gen_range(0..32),
        };
        c.new_weights();
        c
    }
}

impl Creature {
    pub fn feed(&self, input:Vec<f32>) -> Vec<f32> {
        let mut current_layer = input;

        for i in 0..self.brain.len() {
            current_layer = convolution(current_layer,&self.weights[i],&self.brain[i]);
        }

        convolution(current_layer,&self.weights[self.weights.len()-1],&vec![1.0;self.out_size])
    }

    pub fn new_variant(&self) -> Creature {
        let mut rng = rand::thread_rng();
        let mut variant = self.clone();

        let number = rng.gen_range(0..100);
        variant.mutate_node();
        match number {
            1..=10 => variant.add_random_node(),
            //11..=15 => variant.remove_random_node(),
            98 => variant.add_random_layer(),
            90..=97 => variant.new_weights(),
            //99 => variant.remove_random_layer(),
            _ => variant.mutate_node(),
        }
        variant
    }

    pub fn brain_size(&self) -> usize {
        let mut size: usize = 0;
        for layer in &self.brain {
            size += layer.len();
        }
        size
    }

    fn mutate_node(&mut self) {
        let mut rng = rand::thread_rng();
        let layer_i = rng.gen_range(0..self.brain.len());
        let node_i = rng.gen_range(0..self.brain[layer_i].len());
        let modify_brain = rng.gen_range(0.0..1.0) >= 0.5;
        let mut nodes = &self.brain;
        if !modify_brain {
            nodes = &self.weights;
        }
        let mut node = nodes[layer_i][node_i];
        if rng.gen_range(0.0..1.0) >= 0.5 {
            node += node * rng.gen_range(0.0..self.mutate_speed);
        } else {
            node -= node * rng.gen_range(0.0..self.mutate_speed);
        }
        self.brain[layer_i][node_i] = node;
        let change_more = rng.gen_range(0.0..1.0) >= 0.05;
        if change_more { self.mutate_node();}
    }

    fn add_random_layer(&mut self) {
        let mut rng = rand::thread_rng();
        self.brain.push((0..rng.gen_range(1..10)).map(|_| {rng.gen_range(-1.0..1.0)}).collect());
        self.new_weights();
    }

    fn remove_random_layer(&mut self) {
        if self.brain.len() > 1 {
            self.brain.pop();
        }
        self.new_weights();
    }

    fn add_random_node(&mut self) {
        let mut rng = rand::thread_rng();
        let layer_i = rng.gen_range(0..self.brain.len());
        let node_i = rng.gen_range(0..self.brain[layer_i].len());
        self.brain[layer_i].insert(
            node_i,rng.gen_range(-self.mutate_speed..self.mutate_speed)
        );
        let change_more = rng.gen_range(0.0..1.0) >= 0.1;
        if change_more { self.add_random_node();}
        self.new_weights();
    }

    fn remove_random_node(&mut self) {

    }

    fn new_weights(&mut self) {
        let mut rng = rand::thread_rng();
        let mut w: Vec<Vec<f32>> = Vec::with_capacity(self.brain.len());
        let mut last_len = self.in_size;
        for i in 0..self.brain.len() {
            w.push((0..last_len*self.brain[i].len()).map(|_| 
                {
                    rng.gen_range(-1.0..1.0)
                }).collect());
            last_len = self.brain[i].len()
        }
        self.weights = w;
    }
}
