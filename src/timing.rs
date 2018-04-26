use time::{PreciseTime, Duration};
use std::collections::VecDeque;

#[derive(Clone)]
pub enum Timing<'a> {
    Timing {
        name: &'a str,
        time: Duration,
    },
    Block(&'a str, Duration, Box<Timings<'a>>),
}

impl<'a> Timing<'a> {
    fn new(from: &PreciseTime, name: &'a str) -> Timing<'a> {
        Timing::Timing {
            name, time: from.to(PreciseTime::now()),
        }
    }
    fn new_block(from: &PreciseTime, name: &'a str) -> Timing<'a> {
        let now = PreciseTime::now();
        Timing::Block(name, from.to(now), box Timings::new_at(now))
    }

    fn start_time(&self) -> &Duration {
        match self {
            Timing::Timing { time, .. } => time,
            Timing::Block(_, time, _) => time,
        }
    }
}

pub trait Timer<'a> {
    fn time(&mut self, name: &'a str);
    fn block(&mut self, name: &'a str) -> &mut Self;
    fn end_block(&mut self);
}

#[derive(Clone)]
pub struct Timings<'a>(pub PreciseTime, pub VecDeque<Timing<'a>>, bool);
impl<'a> Timings<'a> {
    pub fn new() -> Timings<'a> {
        Timings::new_at(PreciseTime::now())
    }
    pub fn new_at(time: PreciseTime) -> Timings<'a> {
        Timings(time, VecDeque::new(), false)
    }
    pub fn end(mut self) -> Report<'a> {
        if !self.2 { self.end_block(); }
        let mut report = Vec::with_capacity(self.1.len() - 1);

        let mut iter = self.1.into_iter();
        let mut head = iter.next().unwrap();

        for next in iter {
            let item = match head {
                Timing::Timing { time, name } => {
                    let next_start_time = *next.start_time();
                    let delta = next_start_time - time;
                    ReportEntry::Item(name, delta)
                }
                Timing::Block(name, _, block) => {
                    ReportEntry::Block(name, box block.end())
                }
            };
            head = next;
            report.push(item);
        }

        Report(report)
    }
}

impl<'a> Timer<'a> for Timings<'a> {
    fn time(&mut self, name: &'a str) {
        self.1.push_back(Timing::new(&self.0, name))
    }
    fn block(&mut self, name: &'a str) -> &mut Timings<'a> {
        let n = self.1.len();
        self.1.push_back(Timing::new_block(&self.0, name));
        match &mut self.1[n] {
            Timing::Block(_, _, d) => d,
            _ => unimplemented!(),
        }
    }
    fn end_block(&mut self) {
        if self.2 { return; }
        self.time("end");
        self.2 = true;
    }
}

pub struct NoopTimer;
impl<'a> Timer<'a> for NoopTimer {
    fn time(&mut self, _name: &'a str) {}
    fn block(&mut self, _name: &'a str) -> &mut NoopTimer {
        self
    }
    fn end_block(&mut self) {}
}

#[derive(Debug, Clone)]
pub enum ReportEntry<'a> {
    Item(&'a str, Duration),
    Block(&'a str, Box<Report<'a>>),
}

#[derive(Debug, Clone)]
pub struct Report<'a>(Vec<ReportEntry<'a>>);
impl<'a> Report<'a> {
    fn report_prefix(&self, prefix: &str) -> Duration {
        let mut total = Duration::zero();
        let timings = &self.0;
        let num = timings.len();
        for i in 0..num {
            let timing = &timings[i];
            match timing {
                ReportEntry::Item(name, time) => {
                    total = total + *time;
                    let ms = time.num_microseconds().unwrap() as f32 / 1000.0;
                    println!("{}{:040}: {:6.3}ms", prefix, name, ms);
                },
                ReportEntry::Block(name, block) => {
                    println!("{}{}", prefix, name);
                    total = total + block.report_prefix(&(prefix.to_string() + "| "));
                }
            }
        }
        println!(
            "{}{:040}: {:6.3}ms",
            prefix,
            "Total",
            total.num_microseconds().unwrap() as f32 / 1000.0
        );
        total
    }
    pub fn print(&self) {
        println!("\nReport:");
        self.report_prefix("");
    }
    pub fn add(&mut self, other: &Report<'a>) {
        let mut to_add = vec![];
        'other: for (i, t) in other.0.iter().enumerate() {
            if self.0.len() > i {
                let time = &mut self.0[i];
                match (time, t) {
                    (
                        ReportEntry::Item(name, ref mut time),
                        ReportEntry::Item(other_name, other_time)
                    ) => {
                        assert_eq!(*name, *other_name);
                        *time = *time + *other_time;
                    }
                    (
                        ReportEntry::Block(name, ref mut block),
                        ReportEntry::Block(other_name, other_block)
                    ) => {
                        assert_eq!(*name, *other_name);
                        block.add(other_block);
                    }
                    _ => unimplemented!()
                }
            } else {
                let new_time = match t {
                    ReportEntry::Item(name, time) => ReportEntry::Item(name, *time),
                    ReportEntry::Block(name, block) => ReportEntry::Block(name, (*block).clone()),
                };
                to_add.push(new_time);
            }
        }
        for a in to_add.into_iter() {
            self.0.push(a);
        }
    }
    fn normalize_and_divide_all_times(&mut self, by: i32) {
        for t in self.0.iter_mut() {
            match t {
                ReportEntry::Item(_, ref mut time) => {
                    *time = *time / by;
                }
                ReportEntry::Block(_, ref mut block) => {
                    block.normalize_and_divide_all_times(by);
                }
            }
        }
    }
    pub fn averange<T>(timings: T) -> Report<'a>
        where T: Iterator<Item = &'a Report<'a>>
    {
        let mut ts = Report(vec![]);
        let mut n = 0;
        for t in timings {
            ts.add(&t);
            n += 1;
        }
        ts.normalize_and_divide_all_times(n);
        ts
    }
}
