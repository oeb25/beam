use std;

#[derive(Debug, Deserialize)]
pub(crate) struct Contributor {
	pub author: String,
	pub authoring_tool: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AssetUnit {
	pub name: String,
	pub meter: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Asset {
	pub contributor: Contributor,
	pub created: String,
	pub modified: String,
	pub unit: AssetUnit,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryImage {
	pub id: String,
	pub name: String,
	pub init_from: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryImages {
	pub image: Vec<LibraryImage>
}

#[derive(Debug, Deserialize)]
pub(crate) enum NewParamKind {
	#[serde(rename = "surface")]
	Surface {
		#[serde(rename = "type")]
		kind: String,
		init_from: String,
	},
	#[serde(rename = "sampler2D")]
	Sampler2D {
		source: String,
	}
}

#[derive(Debug, Deserialize)]
pub(crate) struct NewParam {
	pub sid: String,
	#[serde(rename = "$value")]
	pub kind: NewParamKind,
}

#[derive(Debug, Deserialize)]
pub(crate) enum TechniqueValue {
	#[serde(rename = "color")]
	Color {
		sid: String,
		#[serde(rename = "$value")]
		value: String,
	},
	#[serde(rename = "texture")]
	Texture {
		texture: String,
	},
	#[serde(rename = "float")]
	Float {
		sid: String,
		#[serde(rename = "$value")]
		value: String,
	},
}

#[derive(Debug, Deserialize)]
pub(crate) enum Technique {
	#[serde(rename = "phong")]
	Phong {
		emission: Option<TechniqueValue>,
		ambient: Option<TechniqueValue>,
		diffuse: Option<TechniqueValue>,
		specular: Option<TechniqueValue>,
        shininess: Option<TechniqueValue>,
		reflective: Option<TechniqueValue>,
        reflectivity: Option<TechniqueValue>,
        transparent: Option<TechniqueValue>,
        transparency: Option<TechniqueValue>,
		index_of_refraction: Option<TechniqueValue>,
	}
}

#[derive(Debug, Deserialize)]
pub(crate) enum LibraryEffectProfileCommonElement {
    #[serde(rename = "newparam")]
    NewParam(NewParam),
    #[serde(rename = "technique")]
    Technique(Technique),
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryEffectProfileCommon {
    #[serde(rename = "$value")]
    pub elements: Vec<LibraryEffectProfileCommonElement>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryEffect {
	pub id: String,
	#[serde(rename = "profile_COMMON")]
	pub profile_common: LibraryEffectProfileCommon,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryEffects {
	pub effect: Vec<LibraryEffect>
}

#[derive(Debug, Deserialize)]
pub(crate) struct InstanceEffect {
	pub url: String
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryMaterial {
	pub id: String,
	pub name: String,
	pub instance_effect: InstanceEffect,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryMaterials {
	pub material: Vec<LibraryMaterial>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Source {
	pub id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct FloatArray {
	pub id: String,
	pub count: String,
	#[serde(rename = "$value", deserialize_with = "deserialize_max")]
	pub data: Vec<f32>,
}

/// Deserialize the maximum of a sequence of values. The entire sequence
/// is not buffered into memory as it would be if we deserialize to Vec<T>
/// and then compute the maximum later.
///
/// This function is generic over T which can be any type that implements
/// Ord. Above, it is used with T=u64.
use std::fmt;
use serde::de::{self, Deserialize, Deserializer, Visitor, SeqAccess};
fn deserialize_max<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de> + std::str::FromStr,
    D: Deserializer<'de>,
    <T as std::str::FromStr>::Err: fmt::Debug
{
    struct MaxVisitor<T>(std::marker::PhantomData<T>);

    impl<'de, T> Visitor<'de> for MaxVisitor<T>
    where
        T: Deserialize<'de> + std::str::FromStr,
        <T as std::str::FromStr>::Err: fmt::Debug
    {
        /// Return type of this visitor. This visitor computes the max of a
        /// sequence of values of type T, so the type of the maximum is T.
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a nonempty sequence of numbers")
        }

        fn visit_seq<S>(self, mut seq: S) -> Result<Vec<T>, S::Error>
        where
            S: SeqAccess<'de>,
        {
        	if let Some(x) = seq.next_element::<String>()? {
        		 let result = x.split_whitespace().map(|x| x.parse().unwrap()).collect();
        		 Ok(result)
        	} else {
        		Err(de::Error::custom("no values in seq when looking for maximum"))
        	}
        }
    }

    let visitor = MaxVisitor(std::marker::PhantomData);
    deserializer.deserialize_seq(visitor)
}

#[derive(Debug, Deserialize)]
pub(crate) struct AccessorParam {
	pub name: String,
	#[serde(rename = "type")]
	pub kind: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Accessor {
	pub source: String,
	pub count: String,
	pub stride: String,
	pub param: Vec<AccessorParam>
}

#[derive(Debug, Deserialize)]
pub(crate) struct MeshSourceTechinqueCommon {
	pub accessor: Accessor,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MeshSource {
	pub id: String,
	pub float_array: FloatArray,
	pub technique_common: MeshSourceTechinqueCommon,
}

#[derive(Debug, Deserialize)]
pub(crate) struct VerticiesInput {
	pub semantic: String,
	pub source: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Verticies {
	pub id: String,
	pub input: VerticiesInput,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TrianglesInput {
	pub semantic: String,
	pub source: String,
	pub offset: String,
	#[serde(default)]
	pub set: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Triangles {
	pub material: String,
	pub count: String,
	pub input: Vec<TrianglesInput>,
	#[serde(deserialize_with = "deserialize_max")]
	pub p: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Mesh {
	pub source: Vec<MeshSource>,
	pub vertices: Verticies,
	pub triangles: Vec<Triangles>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryGeometry {
	pub id: String,
	pub name: String,
	pub mesh: Mesh,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryGeometries {
	pub geometry: Vec<LibraryGeometry>
}

#[derive(Debug, Deserialize)]
pub(crate) struct Matrix {
	pub sid: String,
	#[serde(rename = "$value", deserialize_with = "deserialize_max")]
	pub data: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub(crate) enum BindTechniqueCommon {
	#[serde(rename = "instance_material")]
	InstanceMaterial {
		symbol: String,
		target: String,
	}
}

#[derive(Debug, Deserialize)]
pub(crate) struct BindMaterialTechniqueCommon {
    #[serde(rename = "$value")]
    pub materials: Vec<BindTechniqueCommon>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct BindMaterial {
	pub technique_common: BindMaterialTechniqueCommon,
}

#[derive(Debug, Deserialize)]
pub(crate) struct InstanceGeometry {
	pub url: String,
	pub name: String,
	pub bind_material: Option<BindMaterial>,
}

#[derive(Debug, Deserialize)]
pub(crate) enum SceneNodeKind {
	#[serde(rename = "NODE")]
	Node,
}

#[derive(Debug, Deserialize)]
pub(crate) enum Transformation {
    #[serde(rename = "look_at")]
    LookAt,
    #[serde(rename = "matrix", deserialize_with = "deserialize_max")]
    Matrix(Vec<f32>),
    #[serde(rename = "rotate")]
    Rotate,
    #[serde(rename = "scale")]
    Scale,
    #[serde(rename = "skew")]
    Skew,
    #[serde(rename = "translate")]
    Translate,
}

#[derive(Debug, Deserialize)]
pub(crate) enum SceneNodeElement {
    #[serde(rename = "look_at")]
    LookAt,
    #[serde(rename = "matrix", deserialize_with = "deserialize_max")]
    Matrix(Vec<f32>),
    #[serde(rename = "rotate")]
    Rotate,
    #[serde(rename = "scale")]
    Scale,
    #[serde(rename = "skew")]
    Skew,
    #[serde(rename = "translate")]
    Translate,
    #[serde(rename = "instance_geometry")]
    InstanceGeometry(InstanceGeometry),
}

#[derive(Debug, Deserialize)]
pub(crate) struct SceneNode {
	pub id: String,
	pub name: String,
	#[serde(rename = "type")]
	pub kind: SceneNodeKind,
    #[serde(rename = "$value")]
    pub elements: Vec<SceneNodeElement>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryVisualScene {
	pub id: String,
	pub name: String,
	#[serde(rename = "node")]
	pub nodes: Vec<SceneNode>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct LibraryVisualScenes {
	pub visual_scene: Vec<LibraryVisualScene>
}

#[derive(Debug, Deserialize)]
pub(crate) struct InstanceVisualScene {
	pub url: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Scene {
	pub instance_visual_scene: InstanceVisualScene
}

#[derive(Debug, Deserialize)]
pub(crate) struct ColladaRaw {
	pub asset: Asset,
	pub library_images: LibraryImages,
	pub library_effects: LibraryEffects,
	pub library_materials: LibraryMaterials,
	pub library_geometries: LibraryGeometries,
	pub library_controllers: (),
	pub library_visual_scenes: LibraryVisualScenes,
	pub scene: Scene,
}
