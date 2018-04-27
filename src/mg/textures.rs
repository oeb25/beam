use gl;
use image;
use std::{ptr, mem, os::raw::c_void};

use mg::types::{GlError, GlType};

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum TextureSlot {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
}
impl Into<u32> for TextureSlot {
    fn into(self) -> u32 {
        let i: i32 = self.into();
        gl::TEXTURE0 + i as u32
    }
}
impl Into<i32> for TextureSlot {
    fn into(self) -> i32 {
        use TextureSlot::*;
        match self {
            Zero => 0,
            One => 1,
            Two => 2,
            Three => 3,
            Four => 4,
            Five => 5,
            Six => 6,
            Seven => 7,
            Eight => 8,
            Nine => 9,
            Ten => 10,
        }
    }
}
impl From<usize> for TextureSlot {
    fn from(nth: usize) -> TextureSlot {
        use TextureSlot::*;
        match nth {
            0 => Zero,
            1 => One,
            2 => Two,
            3 => Three,
            4 => Four,
            5 => Five,
            6 => Six,
            7 => Seven,
            8 => Eight,
            9 => Nine,
            10 => Ten,
            x => unimplemented!("{:?}", x),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TextureKind {
    Texture2d,
    CubeMap,
}
impl Into<u32> for TextureKind {
    fn into(self) -> u32 {
        match self {
            TextureKind::Texture2d => gl::TEXTURE_2D,
            TextureKind::CubeMap => gl::TEXTURE_CUBE_MAP,
        }
    }
}

#[derive(Debug)]
pub struct Texture {
    pub id: gl::types::GLuint,
    pub kind: TextureKind,
}

impl Texture {
    pub fn new(kind: TextureKind) -> Texture {
        let id = unsafe {
            let mut tex_id = mem::uninitialized();
            gl::GenTextures(1, &mut tex_id);
            tex_id
        };

        Texture { id, kind }
    }
    pub fn bind(&self) -> TextureBinder {
        TextureBinder::new(self)
    }
    pub fn bind_to(&self, slot: TextureSlot) {
        unsafe {
            gl::ActiveTexture(slot.into());
        }
        self.bind();
        GlError::check().expect(&format!("unable to bind texture to slot {:?}", slot));
    }
}
impl Drop for Texture {
    fn drop(&mut self) {
        unsafe { gl::DeleteTextures(1, &self.id as *const _) }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum TextureTarget {
    Texture2d,
    ProxyTexture2d,
    Texture1dArray,
    ProxyTexture1dArray,
    TextureRectangle,
    ProxyTextureRectangle,
    TextureCubeMapPositiveX,
    TextureCubeMapNegativeX,
    TextureCubeMapPositiveY,
    TextureCubeMapNegativeY,
    TextureCubeMapPositiveZ,
    TextureCubeMapNegativeZ,
    ProxyTextureCubeMap,
}
impl TextureTarget {
    pub fn cubemap_faces() -> [TextureTarget; 6] {
        use TextureTarget::*;
        [
            TextureCubeMapPositiveX,
            TextureCubeMapNegativeX,
            TextureCubeMapPositiveY,
            TextureCubeMapNegativeY,
            TextureCubeMapPositiveZ,
            TextureCubeMapNegativeZ,
        ]
    }
}
impl Into<u32> for TextureTarget {
    fn into(self) -> u32 {
        use TextureTarget::*;
        match self {
            Texture2d => gl::TEXTURE_2D,
            ProxyTexture2d => gl::PROXY_TEXTURE_2D,
            Texture1dArray => gl::TEXTURE_1D_ARRAY,
            ProxyTexture1dArray => gl::PROXY_TEXTURE_1D_ARRAY,
            TextureRectangle => gl::TEXTURE_RECTANGLE,
            ProxyTextureRectangle => gl::PROXY_TEXTURE_RECTANGLE,
            TextureCubeMapPositiveX => gl::TEXTURE_CUBE_MAP_POSITIVE_X,
            TextureCubeMapNegativeX => gl::TEXTURE_CUBE_MAP_NEGATIVE_X,
            TextureCubeMapPositiveY => gl::TEXTURE_CUBE_MAP_POSITIVE_Y,
            TextureCubeMapNegativeY => gl::TEXTURE_CUBE_MAP_NEGATIVE_Y,
            TextureCubeMapPositiveZ => gl::TEXTURE_CUBE_MAP_POSITIVE_Z,
            TextureCubeMapNegativeZ => gl::TEXTURE_CUBE_MAP_NEGATIVE_Z,
            ProxyTextureCubeMap => gl::PROXY_TEXTURE_CUBE_MAP,
        }
    }
}

#[allow(unused)]
pub enum TextureInternalFormat {
    Rgba32f,
    Rgba32i,
    Rgba32ui,
    Rgba16,
    Rgba16f,
    Rgba16i,
    Rgba16ui,
    Rgba,
    Rgba8,
    Rgba8ui,
    Srgb8Alpha8,
    Rgb10A2,
    Rgb10A2ui,
    R11fG11fB10f,
    Rg32f,
    Rg32i,
    Rg32ui,
    Rg16,
    Rg16f,
    Rgb16i,
    Rgb16ui,
    Rg8,
    Rg8i,
    Rg8ui,
    R32f,
    R32i,
    R32ui,
    R16f,
    R16i,
    R16ui,
    R8,
    R8i,
    R8ui,
    Rgba16Snorm,
    Rgba8Snorm,
    Rgb32f,
    Rgb32i,
    Rgb32ui,
    Rgb16Snorm,
    Rgb16f,
    Rgb16,
    Rgb8Snorm,
    Rgb,
    Rgb8,
    Rgb8i,
    Rgb8ui,
    Srgb8,
    Srgb,
    Rgb9E5,
    Rg16Snorm,
    Rg8Snorm,
    CompressedRgRgtc2,
    CompressedSignedRgRgtc2,
    R16Snorm,
    R8Snorm,
    CompressedRedRgtc1,
    CompressedSignedRedRgtc1,
    DepthComponent,
    DepthComponent32f,
    DepthComponent24,
    DepthComponent16,
    Depth32fStencil8,
    Depth24Stencil8,
}

impl Into<u32> for TextureInternalFormat {
    fn into(self) -> u32 {
        use TextureInternalFormat::*;
        match self {
            Rgba32f => gl::RGBA32F,
            Rgba32i => gl::RGBA32I,
            Rgba32ui => gl::RGBA32UI,
            Rgba16 => gl::RGBA16,
            Rgba16f => gl::RGBA16F,
            Rgba16i => gl::RGBA16I,
            Rgba16ui => gl::RGBA16UI,
            Rgba => gl::RGBA,
            Rgba8 => gl::RGBA8,
            Rgba8ui => gl::RGBA8UI,
            Srgb8Alpha8 => gl::SRGB8_ALPHA8,
            Rgb10A2 => gl::RGB10_A2,
            Rgb10A2ui => gl::RGB10_A2UI,
            R11fG11fB10f => gl::R11F_G11F_B10F,
            Rg32f => gl::RG32F,
            Rg32i => gl::RG32I,
            Rg32ui => gl::RG32UI,
            Rg16 => gl::RG16,
            Rg16f => gl::RG16F,
            Rgb16i => gl::RGB16I,
            Rgb16ui => gl::RGB16UI,
            Rg8 => gl::RG8,
            Rg8i => gl::RG8I,
            Rg8ui => gl::RG8UI,
            R32f => gl::R32F,
            R32i => gl::R32I,
            R32ui => gl::R32UI,
            R16f => gl::R16F,
            R16i => gl::R16I,
            R16ui => gl::R16UI,
            R8 => gl::R8,
            R8i => gl::R8I,
            R8ui => gl::R8UI,
            Rgba16Snorm => gl::RGBA16_SNORM,
            Rgba8Snorm => gl::RGBA8_SNORM,
            Rgb32f => gl::RGB32F,
            Rgb32i => gl::RGB32I,
            Rgb32ui => gl::RGB32UI,
            Rgb16Snorm => gl::RGB16_SNORM,
            Rgb16f => gl::RGB16F,
            Rgb16 => gl::RGB16,
            Rgb8Snorm => gl::RGB8_SNORM,
            Rgb => gl::RGB,
            Rgb8 => gl::RGB8,
            Rgb8i => gl::RGB8I,
            Rgb8ui => gl::RGB8UI,
            Srgb => gl::SRGB,
            Srgb8 => gl::SRGB8,
            Rgb9E5 => gl::RGB9_E5,
            Rg16Snorm => gl::RG16_SNORM,
            Rg8Snorm => gl::RG8_SNORM,
            CompressedRgRgtc2 => gl::COMPRESSED_RG_RGTC2,
            CompressedSignedRgRgtc2 => gl::COMPRESSED_SIGNED_RG_RGTC2,
            R16Snorm => gl::R16_SNORM,
            R8Snorm => gl::R8_SNORM,
            CompressedRedRgtc1 => gl::COMPRESSED_RED_RGTC1,
            CompressedSignedRedRgtc1 => gl::COMPRESSED_SIGNED_RED_RGTC1,
            DepthComponent => gl::DEPTH_COMPONENT,
            DepthComponent32f => gl::DEPTH_COMPONENT32F,
            DepthComponent24 => gl::DEPTH_COMPONENT24,
            DepthComponent16 => gl::DEPTH_COMPONENT16,
            Depth32fStencil8 => gl::DEPTH32F_STENCIL8,
            Depth24Stencil8 => gl::DEPTH24_STENCIL8,
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum TextureFormat {
    Red,  // GL_RED,
    Rg,   // GL_RG,
    Rgb,  // GL_RGB,
    Bgr,  // GL_BGR,
    Rgba, // GL_RGBA,
    Bgra, // GL_BGRA,

    DepthComponent,
}
impl Into<u32> for TextureFormat {
    fn into(self) -> u32 {
        use TextureFormat::*;
        match self {
            Red => gl::RED,
            Rg => gl::RG,
            Rgb => gl::RGB,
            Bgr => gl::BGR,
            Rgba => gl::RGBA,
            Bgra => gl::BGRA,
            DepthComponent => gl::DEPTH_COMPONENT,
        }
    }
}

pub struct TextureBinder<'a>(&'a Texture);
impl<'a> TextureBinder<'a> {
    fn new(texture: &Texture) -> TextureBinder {
        unsafe {
            gl::BindTexture(texture.kind.into(), texture.id);
        }
        TextureBinder(texture)
    }
    pub fn empty(
        &self,
        target: TextureTarget,
        level: usize,
        internal_format: TextureInternalFormat,
        width: u32,
        height: u32,
        format: TextureFormat,
        typ: GlType,
    ) -> &TextureBinder {
        unsafe {
            self.image_2d(
                target,
                level,
                internal_format,
                width,
                height,
                format,
                typ,
                ptr::null(),
            );
        }
        self
    }
    pub unsafe fn image_2d(
        &self,
        target: TextureTarget,
        level: usize,
        internal_format: TextureInternalFormat,
        width: u32,
        height: u32,
        format: TextureFormat,
        typ: GlType,
        data: *const c_void,
    ) {
        let internal_format: u32 = internal_format.into();
        gl::TexImage2D(
            target.into(),
            level as i32,
            internal_format as i32,
            width as i32,
            height as i32,
            0,
            format.into(),
            typ.into(),
            data,
        );
    }
    pub fn load_image(
        &self,
        target: TextureTarget,
        internal_format: TextureInternalFormat,
        format: TextureFormat,
        img: &image::DynamicImage,
    ) -> &TextureBinder {
        use image::GenericImage;
        let (w, h) = img.dimensions();

        let data = match format {
            // TextureFormat::Red => img.to_red().into_raw(),
            TextureFormat::Red => unimplemented!(),
            // TextureFormat::Rg => img.to_rg().into_raw(),
            TextureFormat::Rg => unimplemented!(),
            TextureFormat::Rgb => img.to_rgb().into_raw(),
            // TextureFormat::Bgr => img.to_bgr().into_raw(),
            TextureFormat::Bgr => unimplemented!(),
            TextureFormat::Rgba => img.to_rgba().into_raw(),
            // TextureFormat::Bgra => img.to_bgra().into_raw(),
            TextureFormat::Bgra => unimplemented!(),
            TextureFormat::DepthComponent => unimplemented!(),
        };

        unsafe {
            self.image_2d(
                target,
                0,
                internal_format,
                w,
                h,
                format,
                GlType::UnsignedByte,
                &data[0] as *const _ as *const _,
            );
        }
        self
    }
    pub fn parameter_int(&self, pname: TextureParameter, param: i32) -> &TextureBinder {
        unsafe {
            gl::TexParameteri(self.0.kind.into(), pname.into(), param);
        }
        self
    }
}

#[allow(unused)]
pub enum TextureParameter {
    BaseLevel,
    CompareFunc,
    CompareMode,
    LodBias,
    MinFilter,
    MagFilter,
    MinLod,
    MaxLod,
    MaxLevel,
    SwizzleR,
    SwizzleG,
    SwizzleB,
    SwizzleA,
    WrapS,
    WrapT,
    WrapR,
}
impl Into<u32> for TextureParameter {
    fn into(self) -> u32 {
        use TextureParameter::*;
        match self {
            BaseLevel => gl::TEXTURE_BASE_LEVEL,
            CompareFunc => gl::TEXTURE_COMPARE_FUNC,
            CompareMode => gl::TEXTURE_COMPARE_MODE,
            LodBias => gl::TEXTURE_LOD_BIAS,
            MinFilter => gl::TEXTURE_MIN_FILTER,
            MagFilter => gl::TEXTURE_MAG_FILTER,
            MinLod => gl::TEXTURE_MIN_LOD,
            MaxLod => gl::TEXTURE_MAX_LOD,
            MaxLevel => gl::TEXTURE_MAX_LEVEL,
            SwizzleR => gl::TEXTURE_SWIZZLE_R,
            SwizzleG => gl::TEXTURE_SWIZZLE_G,
            SwizzleB => gl::TEXTURE_SWIZZLE_B,
            SwizzleA => gl::TEXTURE_SWIZZLE_A,
            WrapS => gl::TEXTURE_WRAP_S,
            WrapT => gl::TEXTURE_WRAP_T,
            WrapR => gl::TEXTURE_WRAP_R,
        }
    }
}

