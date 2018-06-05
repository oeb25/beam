use mg;
use std::{self, borrow::Cow};
use warmy;

fn rip<T, C>(e: warmy::load::StoreErrorOr<T, C>) -> T::Error
where
    T: warmy::Load<C> + std::fmt::Debug,
{
    match e {
        warmy::load::StoreErrorOr::ResError(e) => e,
        e => panic!("{:?}", e),
    }
}

#[derive(Debug)]
pub struct FromFS {
    pub src: String,
    pub deps: Vec<warmy::DepKey>,
}

impl<C> warmy::Load<C> for FromFS {
    type Key = warmy::FSKey;

    type Error = std::io::Error;

    fn load(
        key: Self::Key,
        storage: &mut warmy::Storage<C>,
        ctx: &mut C,
    ) -> Result<warmy::Loaded<Self>, Self::Error> {
        let path = key.as_path();

        let mut deps = vec![];

        let src: Result<_, std::io::Error> = std::fs::read_to_string(path)?
            .lines()
            .map(|x| {
                let y = x.trim_left();
                let res = if y.starts_with("#include") {
                    let offset = "#include \"".len();
                    let end = y.len() - 1;
                    let req = &y[offset..end];

                    let path = path.strip_prefix(storage.root()).unwrap();

                    // let req = path.parent().unwrap().to_str().unwrap().to_owned() + "/" + req;
                    let req = path.with_file_name(req);
                    let key = warmy::FSKey::new(req);
                    let res = storage.get(&key, ctx).map_err(rip)?;
                    deps.push(key.into());
                    let r: &FromFS = &res.borrow();
                    for d in &r.deps {
                        if !deps.contains(d) {
                            deps.push(d.clone());
                        }
                    }
                    r.src.to_owned()
                } else {
                    x.to_owned()
                };
                Ok(res + "\n")
            })
            .collect();

        let from_fs = FromFS {
            src: src?,
            deps: deps.clone(),
        };

        Ok(warmy::load::Loaded::with_deps(from_fs, deps))
    }
}

#[derive(Clone, Hash)]
pub struct ShaderSrc<'a> {
    pub name: String,
    pub vert: Cow<'a, str>,
    pub geom: Option<Cow<'a, str>>,
    pub frag: Cow<'a, str>,
}

pub struct MyShader {
    pub vert: String,
    pub frag: String,
    pub program: mg::Program,
    pub deps: Vec<warmy::DepKey>,
}

impl std::cmp::PartialEq for MyShader {
    fn eq(&self, rhs: &MyShader) -> bool {
        self.program == rhs.program
    }
}

impl std::fmt::Debug for MyShader {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(fmt, "{:?}", self.program)
    }
}

impl<'a> Into<warmy::key::LogicalKey> for ShaderSrc<'a> {
    fn into(self) -> warmy::key::LogicalKey {
        let mut p = self.name.to_owned() + "," + &self.vert + "," + &self.frag;
        if let Some(q) = self.geom {
            p += ",";
            p += &q;
        }
        warmy::key::LogicalKey::new(p)
    }
}

impl<C> warmy::Load<C> for MyShader {
    type Key = warmy::key::LogicalKey;

    type Error = std::io::Error;

    fn load(
        key: Self::Key,
        storage: &mut warmy::Storage<C>,
        ctx: &mut C,
    ) -> Result<warmy::Loaded<Self>, Self::Error> {
        let mut deps = key.as_str().split(',');
        let name = deps.next().unwrap();
        let vert = deps.next().unwrap();
        let frag = deps.next().unwrap();
        let geom = deps.next();

        let vert_key = warmy::FSKey::new(vert);
        let frag_key = warmy::FSKey::new(frag);
        let geom_key = geom.map(warmy::FSKey::new);

        let vert_src: warmy::Res<FromFS> = storage.get(&vert_key, ctx).map_err(rip)?;
        let frag_src: warmy::Res<FromFS> = storage.get(&frag_key, ctx).map_err(rip)?;
        let geom_src: Option<warmy::Res<FromFS>> = geom_key
            .map(|geom_key| storage.get(&geom_key, ctx))
            .transpose()
            .map_err(rip)?;
        let vert = &vert_src.borrow();
        let frag = &frag_src.borrow();
        let geom = geom_src.as_ref().map(|geom_src| geom_src.borrow());
        let mut deps = vec![vert_key.into(), frag_key.into()];
        for d in vert.deps.iter().chain(frag.deps.iter()) {
            if !deps.contains(d) {
                deps.push(d.clone());
            }
        }
        if let Some(geom) = &geom {
            for d in &geom.deps {
                if !deps.contains(d) {
                    deps.push(d.clone());
                }
            }
        }

        let version = "410 core";

        let vert_src = format!("#version {}\n{}", version, &vert.src);
        let frag_src = format!("#version {}\n{}", version, &frag.src);
        let geom_src = geom.map(|geom| format!("#version {}\n{}", version, &geom.src));

        let mut program = mg::Program::new_from_src(
            &vert_src,
            match &geom_src {
                Some(x) => Some(x),
                None => None,
            },
            &frag_src,
        ).expect("unable to create program");

        program.set_name(name.to_owned());

        let res = MyShader {
            vert: vert_src,
            frag: frag_src,
            program,
            deps: deps.clone(),
        };

        Ok(warmy::Loaded::with_deps(res, deps))
    }
}
