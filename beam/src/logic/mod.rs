use std::ops;
use cgmath::Rad;
use misc::{v3, Mat4};
use render;
use std::f32::consts::PI;

const MAP_SIZE: (usize, usize) = (10, 10);
type Map = [[Tile; MAP_SIZE.1]; MAP_SIZE.0];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tile {
    Empty,
    Wall,
}

impl Default for Tile {
    fn default() -> Tile {
        Tile::Empty
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Move(Direction),
    Undo
}

#[allow(unused)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Right,
    Down,
    Left,
}

impl Direction {
    pub fn to_tuple(&self) -> (i32, i32) {
        use self::Direction::*;
        match self {
            Up => (0, 1),
            Right => (1, 0),
            Down => (0, -1),
            Left => (-1, 0),
        }
    }
    pub fn to_rotation(&self) -> Mat4 {
        use self::Direction::*;
        let frac = match self {
            Left => 1.0 / 4.0,
            Down => 2.0 / 4.0,
            Right => 3.0 / 4.0,
            Up => 4.0 / 4.0,
        };
        Mat4::from_angle_y(Rad(frac * 2.0 * PI))
    }
    pub fn clockwise(&self) -> Direction {
        use self::Direction::*;
        match self {
            Up => Right,
            Right => Down,
            Down => Left,
            Left => Up,
        }
    }
    pub fn couter_clockwise(&self) -> Direction {
        use self::Direction::*;
        match self {
            Up => Left,
            Right => Up,
            Down => Right,
            Left => Down,
        }
    }
    pub fn opposite(&self) -> Direction {
        self.clockwise().clockwise()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    Owl,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    pub pos: (i32, i32),
    pub kind: EntityKind,
    pub direction: Direction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LastEntityState {
    pos: (i32, i32),
    direction: Direction,
}

fn lerp<T>(a: T, t: f32, b: T) -> T
where
    T: ops::Add<Output = T> + ops::Sub<Output = T> + Copy,
    f32: ops::Mul<T, Output = T>,
{
    a + t * (b - a)
}

impl Entity {
    fn animate(&self, last: &LastEntityState, t: f32) -> Mat4 {
        let tt = t.min(1.0);

        match self.kind {
            EntityKind::Owl => {
                let a = v3(last.pos.0 as f32, 0.5, last.pos.1 as f32);
                let b = v3(self.pos.0 as f32, 0.5, self.pos.1 as f32);

                let mut pos = lerp(a, tt, b);

                pos.y += (tt * PI).sin();
                if t > 1.0 {
                    pos.y += (t * PI * 2.0).sin().abs() * (1.0 / (t + 0.5)).powf(4.0) * 1.5;
                }

                Mat4::from_translation(pos) * self.direction.to_rotation()
            }
        }
    }
}

impl<'a> Into<LastEntityState> for Entity {
    fn into(self) -> LastEntityState {
        LastEntityState {
            pos: self.pos,
            direction: self.direction,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Game {
    pub map: Map,
    pub owl: (LastEntityState, Entity),
}

impl Game {
    pub fn new() -> Game {
        let owl = Entity {
            pos: (5, 4),
            kind: EntityKind::Owl,
            direction: Direction::Up,
        };
        let mut game = Game {
            map: Default::default(),
            owl: (owl.clone().into(), owl),
        };
        for ((x, y), tile) in game.tiles_mut() {
            let is_on_x_bounds = x == 0 || x == MAP_SIZE.0 - 1;
            let is_on_y_bounds = y == 0 || y == MAP_SIZE.1 - 1;
            if is_on_x_bounds || is_on_y_bounds {
                *tile = Tile::Wall;
            } else if (x + 2 * y) % 4 == 0 {
                *tile = Tile::Wall;
            }
        }
        game
    }
    pub fn action(&self, action: Action) -> Game {
        let mut game = self.clone();

        match action {
            Action::Move(direction) => {
                game.owl.1.direction = direction;

                let move_by = direction.to_tuple();

                let new_pos = (self.owl.1.pos.0 + move_by.0, self.owl.1.pos.1 + move_by.1);

                game.owl.0 = game.owl.1.clone().into();

                if game.map[new_pos.0 as usize][new_pos.1 as usize] == Tile::Empty {
                    game.owl.1.pos = new_pos;
                }

                game
            },
            Action::Undo => game,
        }
    }
    pub fn tiles_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut Tile)> {
        self.map.iter_mut().enumerate().flat_map(|(x, c)| {
            c.iter_mut().enumerate().map(move |(y, tile)| {
                ((x, y), tile)
            })
        })
    }
    pub fn tiles(&self) -> impl Iterator<Item = ((usize, usize), &Tile)> {
        self.map.iter().enumerate().flat_map(|(x, c)| {
            c.iter().enumerate().map(move |(y, tile)| {
                ((x, y), tile)
            })
        })
    }
    pub fn render(
        &self,
        props: &RenderProps,
        t: f32,
    ) -> Vec<render::RenderObject> {
        let mut calls = vec![];

        for ((x, y), tile) in self.tiles() {
            let pos = v3(MAP_SIZE.0 as f32 - 1.0 - x as f32, 0.0, y as f32);
            match tile {
                Tile::Empty => {
                    calls.push(props.cube_mesh.translate(pos));
                }
                Tile::Wall => {
                    let wall = props
                        .cube_mesh
                        .scale_nonuniformly(v3(1.0, 2.0, 1.0))
                        .translate(pos);

                    calls.push(wall)
                }
            }
        }

        let mut owl_pos = self.owl.1.animate(&self.owl.0, t);
        owl_pos[3][0] = MAP_SIZE.0 as f32 - 1.0 - owl_pos[3][0];
        let owl = props.owl_mesh.scale(0.3).transform(owl_pos);

        calls.push(owl);

        calls
    }
}

#[derive(Debug, Clone)]
pub struct RenderProps<'a> {
    pub owl_mesh: &'a render::RenderObject,
    pub cube_mesh: &'a render::RenderObject,
    pub plastic_material: render::Material,
}
