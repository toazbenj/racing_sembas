pub trait Queue<T> {
    fn enqueue(&mut self, x: T);
    fn dequeue(&mut self) -> Option<T>;
}

impl<T> Queue<T> for Vec<T> {
    fn enqueue(&mut self, x: T) {
        self.push(x);
    }

    fn dequeue(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            Some(self.remove(0))
        }
    }
}
