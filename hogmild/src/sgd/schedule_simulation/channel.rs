use crossbeam::channel;

use super::*;

struct Receiver<T, N>
where
    T: HasTime,
    N: Node,
{
    node: N,
    recv: channel::Receiver<T>,
}

impl<T, N> Receiver<T, N>
where
    T: HasTime,
    N: Node,
{
    fn next_time(&self) -> Option<Tick> {
        self.recv.try_iter().peekable().peek().map(|p| p.time())
    }
}

struct Sender<T, N>
where
    T: HasTime,
    N: Node,
{
    node: N,
    recv: channel::Receiver<T>,
}

impl<T, N> Sender<T, N>
where
    T: HasTime,
    N: Node,
{
}
