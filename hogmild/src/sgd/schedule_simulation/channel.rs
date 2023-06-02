use crossbeam::channel;

use super::*;

struct Receiver<'a, T, N>
where
    T: HasTime,
    N: Node,
{
    node: &'a N,
    rx: channel::Receiver<T>,
    next_packet: Option<T>,
    /// Time when an element is dequeued
    ack_tx: channel::Sender<Tick>,
}

impl<'a, T, N> Receiver<'a, T, N>
where
    T: HasTime,
    N: Node,
{
    fn new(node: &'a N, rx: channel::Receiver<T>, ack_tx: channel::Sender<Tick>) -> Self {
        Self {
            node,
            rx,
            next_packet: None,
            ack_tx,
        }
    }

    pub fn next_time(&mut self) -> Option<Tick> {
        match &self.next_packet {
            Some(p) => Some(p.time()),
            None => {
                self.next_packet = self.rx.try_recv().ok();
                self.next_packet.as_ref().map(|p| p.time())
            }
        }
    }
}

struct Sender<'a, T, N>
where
    T: HasTime,
    N: Node,
{
    node: &'a N,
    tx: channel::Sender<T>,
    /// Time when an element is dequeued
    ack_rx: channel::Receiver<Tick>,
    next_ack: Option<Tick>,
    cap: usize,
    len: usize,
}

impl<'a, T, N> Sender<'a, T, N>
where
    T: HasTime,
    N: Node,
{
    fn new(
        cap: usize,
        node: &'a N,
        tx: channel::Sender<T>,
        ack_rx: channel::Receiver<Tick>,
    ) -> Self {
        Self {
            node,
            tx,
            ack_rx,
            next_ack: None,
            cap,
            len: 0,
        }
    }

    /// Peek the next ack without consuming it.
    fn peek_ack(&mut self) -> Option<&Tick> {
        match self.next_ack {
            Some(_) => self.next_ack.as_ref(),
            None => {
                self.next_ack = self.ack_rx.recv().ok();
                self.next_ack.as_ref()
            }
        }
    }

    /// Caller must ensure `next_ack` is not None
    fn consume_ack(&mut self) {
        assert!(self.next_ack.is_some());
        self.next_ack = None;
    }

    fn update_len(&mut self) {
        while let Some(&t) = self.peek_ack() {
            if self.node.time() >= t {
                self.consume_ack();
                self.len -= 1;
            } else {
                return;
            }
        }
    }

    fn is_full(&mut self) -> bool {
        self.update_len();
        self.len == self.cap
    }

    fn send(&mut self, packet: T) -> Result<(), channel::SendError<T>> {
        self.update_len();
        if self.is_full() {
            Err(channel::SendError(packet))
        } else {
            self.tx.send(packet)
        }
    }
}

fn bounded<'a, T, SN, RN>(
    cap: usize,
    sender_node: &'a SN,
    receiver_node: &'a RN,
) -> (Sender<'a, T, SN>, Receiver<'a, T, RN>)
where
    T: HasTime,
    SN: Node,
    RN: Node,
{
    let (tx, rx) = channel::bounded(cap);
    let (ack_tx, ack_rx) = channel::bounded(cap);
    let send = Sender::new(cap, sender_node, tx, ack_rx);
    let recv = Receiver::new(receiver_node, rx, ack_tx);
    (send, recv)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleNode {
        tick: Tick,
    }

    impl HasTime for SimpleNode {
        fn time(&self) -> Tick {
            self.tick
        }
    }

    impl Node for SimpleNode {
        fn advance_to_time(&mut self, time: Tick) {
            self.tick = max(self.tick, time);
        }
    }

    #[test]
    fn test_channels() {
        let sender_node = SimpleNode { tick: 0 };
        let receiver_node = SimpleNode { tick: 0 };
        let (tx, rx) = bounded::<Tick, _, _>(2, &sender_node, &receiver_node);

        unimplemented!()
    }
}
