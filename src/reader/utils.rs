
#[derive(Default, Debug)]
pub(crate) struct OneCache<T: PartialEq + Eq, U> {
    last_key: Option<T>,
    last_value: Option<U>,
}

impl<T: PartialEq + Eq, U> OneCache<T, U> {
    pub(crate) fn get<F: FnOnce() -> U>(&mut self, key: T, callback: F) -> &U {
        let key = Some(key);
        if self.last_key == key {
            return self.last_value.as_ref().unwrap();
        } else {
            self.last_key = key;
            self.last_value = Some(callback());
            return self.last_value.as_ref().unwrap();
        }
    }
}
