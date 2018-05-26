#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cacher<K: PartialEq, V> {
    cache: Vec<(K, V)>,
}
impl<K: PartialEq, V> Cacher<K, V> {
    pub fn new() -> Cacher<K, V> {
        Cacher {
            cache: vec![],
        }
    }
    pub fn get<'a>(&'a self, key: &K) -> Option<&'a V> {
        self.cache.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.cache.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    // pub fn get_or_else<'a, F>(&'a mut self, key: K, f: F) -> &'a V
    // where
    //     F: FnOnce() -> V
    // {
    //     if let Some(value) = self.get(&key) {
    //         return value;
    //     } else {
    //         let value = f();
    //         self.insert(key, value)
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
        Cacher {
            cache: vec![],
        }
    }
}
