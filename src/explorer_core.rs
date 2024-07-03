use crate::structs::Halfspace;

trait Explorer<const N: usize> {
    fn step(&mut self);

    fn get_boundary(&self) -> &Vec<Halfspace<N>>;
    fn get_boundary_count(&self) -> u32;
}
