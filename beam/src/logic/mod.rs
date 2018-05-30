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
    Up,
    Right,
    Down,
    Left,
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
    fn to_tuple(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, 1),
            Direction::Right => (1, 0),
            Direction::Down => (0, -1),
            Direction::Left => (-1, 0),
        }
    }
    fn to_rotation(&self) -> Mat4 {
        let frac = match self {
            Direction::Left => 1.0 / 4.0,
            Direction::Down => 2.0 / 4.0,
            Direction::Right => 3.0 / 4.0,
            Direction::Up => 4.0 / 4.0,
        };
        Mat4::from_angle_y(Rad(frac * 2.0 * PI))
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

impl Entity {
    #[allow(unused)]
    fn animate(&self, last: &LastEntityState, t: f32) -> Mat4 {
        match self.kind {
            EntityKind::Owl => {
                Mat4::from_translation(v3(
                    MAP_SIZE.0 as f32 - 1.0 - (self.pos.0 as f32),
                    0.5,
                    self.pos.1 as f32,
                )) * self.direction.to_rotation()
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
            pos: (4, 4),
            kind: EntityKind::Owl,
            direction: Direction::Up,
        };
        let mut game = Game {
            map: Default::default(),
            owl: (owl.clone().into(), owl),
        };
        game.map_all_tiles_mut(|(x, y), tile| {
            let is_on_x_bounds = x == 0 || x == MAP_SIZE.0 - 1;
            let is_on_y_bounds = y == 0 || y == MAP_SIZE.1 - 1;
            if is_on_x_bounds || is_on_y_bounds {
                *tile = Tile::Wall;
            }
        });
        game
    }
    pub fn action(&self, action: Action) -> Game {
        let mut game = self.clone();

        let direction = match action {
            Action::Up => Direction::Up,
            Action::Right => Direction::Right,
            Action::Down => Direction::Down,
            Action::Left => Direction::Left,
        };

        game.owl.1.direction = direction;

        let move_by = direction.to_tuple();

        let new_pos = (self.owl.1.pos.0 + move_by.0, self.owl.1.pos.1 + move_by.1);

        if game.map[new_pos.0 as usize][new_pos.1 as usize] == Tile::Empty {
            game.owl.1.pos = new_pos;
        }
        game.owl.0 = game.owl.1.clone().into();

        game
    }
    #[allow(unused)]
    pub fn map_all_tiles<F>(&self, f: F) -> Game
    where
        F: Fn((usize, usize), &Tile) -> Tile,
    {
        let mut new_map: Map = Default::default();

        for (x, c) in self.map.iter().enumerate() {
            for (y, tile) in c.iter().enumerate() {
                new_map[x][y] = f((x, y), tile);
            }
        }

        Game {
            map: new_map,
            owl: self.owl.clone(),
        }
    }
    pub fn map_all_tiles_mut<F>(&mut self, f: F)
    where
        F: Fn((usize, usize), &mut Tile),
    {
        for (x, c) in self.map.iter_mut().enumerate() {
            for (y, tile) in c.iter_mut().enumerate() {
                f((x, y), tile);
            }
        }
    }
    pub fn render(
        &self,
        owl_mesh: &render::RenderObject,
        meshes: &mut render::MeshStore,
        t: f32,
    ) -> Vec<render::RenderObject> {
        let mut calls = vec![];
        let cube_mesh = meshes.get_cube();
        let cube = render::RenderObject::mesh(cube_mesh);

        for (x, c) in self.map.iter().enumerate() {
            for (y, tile) in c.iter().enumerate() {
                let pos = v3(x as f32, 0.0, y as f32);
                match tile {
                    Tile::Empty => {
                        calls.push(cube.translate(pos));
                    }
                    Tile::Wall => {
                        calls.push(cube.scale_nonuniformly(v3(1.0, 2.0, 1.0)).translate(pos))
                    }
                }
            }
        }

        let owl = owl_mesh.transform(self.owl.1.animate(&self.owl.0, t)).scale(0.3);

        calls.push(owl);

        calls
    }
}
