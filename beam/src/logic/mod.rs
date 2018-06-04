use cgmath::Rad;
use misc::{v3, Mat4};
use render;
use std::f32::consts::PI;
use std::ops;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pos(pub i32, pub i32);

impl ops::Add for Pos {
    type Output = Pos;
    fn add(self, other: Pos) -> Pos {
        Pos(self.0 + other.0, self.1 + other.1)
    }
}

impl ops::AddAssign for Pos {
    fn add_assign(&mut self, other: Pos) {
        self.0 += other.0;
        self.1 += other.1;
    }
}

impl ops::Mul<i32> for Pos {
    type Output = Pos;
    fn mul(self, scalar: i32) -> Pos {
        Pos(self.0 * scalar, self.1 * scalar)
    }
}

impl ops::Mul<Pos> for i32 {
    type Output = Pos;
    fn mul(self, pos: Pos) -> Pos {
        Pos(self * pos.0, self * pos.1)
    }
}

const MAP_SIZE: (usize, usize) = (10, 10);

#[derive(Debug, Clone)]
pub struct MapRow([Tile; MAP_SIZE.1]);

#[derive(Debug, Clone)]
pub struct Map([MapRow; MAP_SIZE.0]);

lazy_static! {
    static ref EMPTY_ROW: MapRow = MapRow::default();
}

impl ops::Index<Pos> for Map {
    type Output = Tile;

    fn index(&self, i: Pos) -> &Tile {
        &self[i.0][i.1]
    }
}

impl ops::Index<i32> for Map {
    type Output = MapRow;

    fn index(&self, i: i32) -> &MapRow {
        if (i as usize) < self.0.len() && i > 0 {
            return &self.0[i as usize];
        }

        &EMPTY_ROW
    }
}

impl Default for Map {
    fn default() -> Map {
        Map(Default::default())
    }
}

impl ops::Index<i32> for MapRow {
    type Output = Tile;

    fn index(&self, i: i32) -> &Tile {
        if (i as usize) < self.0.len() && i > 0 {
            return &self.0[i as usize];
        }

        &Tile::Empty
    }
}

impl Default for MapRow {
    fn default() -> MapRow {
        MapRow(Default::default())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Tile {
    Empty,
    Floor,
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
    Undo,
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
    pub fn to_pos(&self) -> Pos {
        use self::Direction::*;
        match self {
            Up => Pos(0, 1),
            Right => Pos(1, 0),
            Down => Pos(0, -1),
            Left => Pos(-1, 0),
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
    Kangaroo,
    Penguin,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    pub pos: Pos,
    pub kind: EntityKind,
    pub direction: Direction,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LastEntityState {
    pos: Pos,
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
            EntityKind::Kangaroo => {
                let a = v3(last.pos.0 as f32, 0.5, last.pos.1 as f32);
                let b = v3(self.pos.0 as f32, 0.5, self.pos.1 as f32);

                let mut pos = lerp(a, tt, b);

                pos.y += (tt * PI).sin() * 1.5;
                if t > 1.0 {
                    pos.y += (t * PI * 2.0).sin().abs() * (1.0 / (t + 0.5)).powf(4.0) * 1.5;
                }

                Mat4::from_translation(pos) * self.direction.to_rotation()
            }
            EntityKind::Penguin => {
                let a = v3(last.pos.0 as f32, 0.5, last.pos.1 as f32);
                let b = v3(self.pos.0 as f32, 0.5, self.pos.1 as f32);

                let mut pos = lerp(a, tt * tt, b);

                pos.y += (tt * PI).sin() * -0.2;

                Mat4::from_translation(pos) * self.direction.to_rotation()
            }
        }
    }
}

impl Into<LastEntityState> for Entity {
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
    pub entities: Vec<(LastEntityState, Entity)>,
}

impl Game {
    pub fn new() -> Game {
        let owl = Entity {
            pos: Pos(5, 4),
            kind: EntityKind::Owl,
            direction: Direction::Up,
        };
        let mut game = Game {
            map: Default::default(),
            entities: vec![(owl.clone().into(), owl)],
        };
        for ((x, y), tile) in game.tiles_mut() {
            let is_on_x_bounds = x == 0 || x == MAP_SIZE.0 - 1;
            let is_on_y_bounds = y == 0 || y == MAP_SIZE.1 - 1;
            if is_on_x_bounds || is_on_y_bounds {
                *tile = Tile::Wall;
            } else if (x + 2 * y) % 4 == 0 {
                *tile = Tile::Wall;
            } else {
                *tile = Tile::Floor;
            }
        }
        game
    }
    pub fn action(&self, action: Action) -> Game {
        let mut game = self.clone();

        match action {
            Action::Move(direction) => {
                for subject in &mut game.entities {
                    subject.1.direction = direction;

                    subject.0 = subject.1.clone().into();

                    let entity = &mut subject.1;

                    let move_by = direction.to_pos();

                    match entity.kind {
                        EntityKind::Owl => {
                            let new_pos = entity.pos + move_by;

                            if game.map[new_pos] == Tile::Floor {
                                entity.pos = new_pos;
                            }
                        }
                        EntityKind::Kangaroo => {
                            for i in (1..=2).rev() {
                                let new_pos = entity.pos + i * move_by;
                                if game.map[new_pos] == Tile::Floor {
                                    entity.pos = new_pos;
                                    break;
                                }
                            }
                        }
                        EntityKind::Penguin => {
                            for i in 1..10 {
                                let new_pos = entity.pos + i * move_by;

                                if game.map[new_pos] != Tile::Floor {
                                    entity.pos += (i - 1) * move_by;

                                    break;
                                }
                            }
                        }
                    }
                }

                game
            }
            Action::Undo => game,
        }
    }
    pub fn tiles_mut(&mut self) -> impl Iterator<Item = ((usize, usize), &mut Tile)> {
        self.map.0.iter_mut().enumerate().flat_map(|(x, c)| {
            c.0
                .iter_mut()
                .enumerate()
                .map(move |(y, tile)| ((x, y), tile))
        })
    }
    pub fn tiles(&self) -> impl Iterator<Item = ((usize, usize), &Tile)> {
        self.map
            .0
            .iter()
            .enumerate()
            .flat_map(|(x, c)| c.0.iter().enumerate().map(move |(y, tile)| ((x, y), tile)))
    }
    pub fn render(&self, props: &RenderProps, t: f32) -> Vec<render::RenderObject> {
        let mut calls = vec![];

        for ((x, y), tile) in self.tiles() {
            let pos = v3(MAP_SIZE.0 as f32 - 1.0 - x as f32, 0.0, y as f32);
            match tile {
                Tile::Empty => {}
                Tile::Floor => {
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

        for entity in &self.entities {
            let mut position = entity.1.animate(&entity.0, t);
            position[3][0] = MAP_SIZE.0 as f32 - 1.0 - position[3][0];
            match entity.1.kind {
                EntityKind::Owl => {
                    let draw_call = props.owl_mesh.scale(0.3).transform(position);
                    calls.push(draw_call);
                }
                EntityKind::Kangaroo => {
                    let new_material = render::Material::new().albedo(v3(0.6, 0.2, 0.3));
                    let draw_call = props
                        .owl_mesh
                        .with_material(new_material)
                        .scale(0.3)
                        .transform(position);
                    calls.push(draw_call);
                }
                EntityKind::Penguin => {
                    let new_material = render::Material::new()
                        .albedo(v3(0.3, 0.3, 0.9))
                        .roughness(0.2)
                        .metallic(0.5);
                    let draw_call = props
                        .owl_mesh
                        .with_material(new_material)
                        .scale(0.3)
                        .transform(position);
                    calls.push(draw_call);
                }
            }
        }

        calls
    }
}

#[derive(Debug, Clone)]
pub struct RenderProps<'a> {
    pub owl_mesh: &'a render::RenderObject,
    pub cube_mesh: &'a render::RenderObject,
    pub plastic_material: render::Material,
}
