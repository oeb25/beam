use cgmath;

pub type V2 = cgmath::Vector2<f32>;
pub type V3 = cgmath::Vector3<f32>;
pub type V4 = cgmath::Vector4<f32>;
pub type P3 = cgmath::Point3<f32>;
#[allow(unused)]
pub type Mat3 = cgmath::Matrix3<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;

pub fn v2(x: f32, y: f32) -> V2 {
    V2::new(x, y)
}

pub fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3::new(x, y, z)
}

pub fn v4(x: f32, y: f32, z: f32, w: f32) -> V4 {
    V4::new(x, y, z, w)
}

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub pos: V3,
    pub norm: V3,
    pub tex: V2,
    pub tangent: V3,
    // pub bitangent: V3,
}

impl Default for Vertex {
    fn default() -> Vertex {
        let zero3 = v3(0.0, 0.0, 0.0);
        Vertex {
            pos: zero3,
            norm: zero3,
            tex: v2(0.0, 0.0),
            tangent: zero3,
            // bitangent: zero3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cacher<K: PartialEq, V> {
    cache: Vec<(K, V)>,
}
impl<K: PartialEq, V> Cacher<K, V> {
    pub fn new() -> Cacher<K, V> {
        Cacher { cache: vec![] }
    }
    pub fn get<'a>(&'a self, key: &K) -> Option<&'a V> {
        self.cache.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.cache
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
    // pub fn get_or_else<'a, F>(&'a mut self, key: K, f: F) -> &'a V
    // where
    //     F: FnOnce() -> V
    // {
    //     match self.get(&key) {
    //         Some(value) => value,
    //         None => self.insert(key, f())
    //     }
    // }
    pub fn insert<'a>(&'a mut self, key: K, value: V) -> &'a V {
        let index = self.cache.len();
        self.cache.push((key, value));
        &self.cache[index].1
    }
    pub fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.cache.iter()
    }
    pub fn into_iter(self) -> impl Iterator<Item = (K, V)> {
        self.cache.into_iter()
    }
}
impl<K: PartialEq, V> Cacher<K, Vec<V>> {
    pub fn push_into(&mut self, key: K, value: V) {
        if let Some(list) = self.get_mut(&key) {
            list.push(value);
        } else {
            self.insert(key, vec![value]);
        }
    }
}
impl<K: PartialEq, V> Default for Cacher<K, V> {
    fn default() -> Cacher<K, V> {
        Cacher { cache: vec![] }
    }
}
