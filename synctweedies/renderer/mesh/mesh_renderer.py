import torch

from pytorch3d.io import load_objs_as_meshes, save_obj, IO

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturesUV
)

from synctweedies.renderer.mesh.geometry import HardGeometryShader
from synctweedies.renderer.mesh.shader import HardNChannelFlatShader
from synctweedies.renderer.mesh.voronoi import voronoi_solve


class UVProjection():
    def __init__(self, texture_size=96, render_size=64, sampling_mode="nearest", channels=3, device=None):
        self.channels = channels
        self.device = device or torch.device("cpu")
        self.lights = AmbientLights(ambient_color=((1.0,)*channels,), device=self.device)
        self.target_size = (texture_size,texture_size)
        self.render_size = render_size
        self.sampling_mode = sampling_mode


    def load_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
        mesh = load_objs_as_meshes([mesh_path], device=self.device)
        if auto_center:
            verts = mesh.verts_packed()
            faces = mesh.faces_packed()
            center = verts.mean(dim=0)
            verts = verts - center
            scale = torch.max(torch.norm(verts, p=2, dim=1))
            verts = verts / scale
            verts *= scale_factor
            mesh = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0), textures=mesh.textures)

        else:
            mesh.scale_verts_((scale_factor))

        if autouv or (mesh.textures is None):
            print("Auto uv using Xatlas")
            mesh = self.uv_unwrap(mesh)

        self.mesh = mesh


    def load_glb_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
        from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
        io = IO()
        io.register_meshes_format(MeshGlbFormat())
        with open(mesh_path, "rb") as f:
            mesh = io.load_mesh(f, include_textures=True, device=self.device)
        if auto_center:
            verts = mesh.verts_packed()
            faces = mesh.faces_packed()
            center = verts.mean(dim=0)
            verts = verts - center
            scale = torch.max(torch.norm(verts, p=2, dim=1))
            verts = verts / scale
            verts *= scale_factor 
            mesh = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0), textures=mesh.textures)

        else:
            mesh.scale_verts_((scale_factor))
        if autouv or (mesh.textures is None):
            mesh = self.uv_unwrap(mesh)
        self.mesh = mesh


    def load_ply_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=True):
        io = IO()
        with open(mesh_path, "rb") as f:
            mesh = io.load_mesh(f, device=self.device)
        if auto_center:
            verts = mesh.verts_packed()
            max_bb = (verts - 0).max(0)[0]
            min_bb = (verts - 0).min(0)[0]
            scale = (max_bb - min_bb).max()/2 
            center = (max_bb+min_bb) /2
            mesh.offset_verts_(-center)
            mesh.scale_verts_((scale_factor / float(scale)))
        else:
            mesh.scale_verts_((scale_factor))

        if autouv:
            mesh = self.uv_unwrap(mesh)
        
        self.mesh = mesh



    def save_mesh(self, mesh_path, texture):
        save_obj(mesh_path, 
                self.mesh.verts_list()[0],
                self.mesh.faces_list()[0],
                verts_uvs= self.mesh.textures.verts_uvs_list()[0],
                faces_uvs= self.mesh.textures.faces_uvs_list()[0],
                texture_map=texture)

    def uv_unwrap(self, mesh):
        verts_list = mesh.verts_list()[0]
        faces_list = mesh.faces_list()[0]


        import xatlas
        import numpy as np
        v_np = verts_list.cpu().numpy()
        f_np = faces_list.int().cpu().numpy()
        atlas = xatlas.Atlas()
        atlas.add_mesh(v_np, f_np)
        chart_options = xatlas.ChartOptions()
        chart_options.max_iterations = 4
        atlas.generate(chart_options=chart_options)
        vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

        vt = torch.from_numpy(vt_np.astype(np.float32)).type(verts_list.dtype).to(mesh.device)
        ft = torch.from_numpy(ft_np.astype(np.int64)).type(faces_list.dtype).to(mesh.device)

        new_map = torch.zeros(self.target_size+(self.channels,), device=mesh.device)
        new_tex = TexturesUV(
            [new_map], 
            [ft], 
            [vt], 
            sampling_mode=self.sampling_mode
            )

        mesh.textures = new_tex
        return mesh


    def disconnect_faces(self):
        mesh = self.mesh
        verts_list = mesh.verts_list()
        faces_list = mesh.faces_list()
        verts_uvs_list = mesh.textures.verts_uvs_list()
        faces_uvs_list = mesh.textures.faces_uvs_list()
        packed_list = [v[f] for v,f in zip(verts_list, faces_list)]
        verts_disconnect_list = [
            torch.zeros(
                (verts_uvs_list[i].shape[0], 3), 
                dtype=verts_list[0].dtype, 
                device=verts_list[0].device
            ) 
            for i in range(len(verts_list))]
        for i in range(len(verts_list)):
            verts_disconnect_list[i][faces_uvs_list] = packed_list[i]
        assert not mesh.has_verts_normals(), "Not implemented for vertex normals"
        self.mesh_d = Meshes(verts_disconnect_list, faces_uvs_list, mesh.textures)
        return self.mesh_d

    def construct_uv_mesh(self):
        mesh = self.mesh_d
        verts_list = mesh.verts_list()
        verts_uvs_list = mesh.textures.verts_uvs_list()
        new_verts_list = []
        for i, (verts, verts_uv) in enumerate(zip(verts_list, verts_uvs_list)):
            verts = verts.clone()
            verts_uv = verts_uv.clone()
            verts[...,0:2] = verts_uv[...,:]
            verts = (verts - 0.5) * 2
            verts[...,2] *= 1
            new_verts_list.append(verts)
        textures_uv = mesh.textures.clone()
        self.mesh_uv = Meshes(new_verts_list, mesh.faces_list(), textures_uv)
        return self.mesh_uv


    def set_texture_map(self, texture):
        new_map = texture.permute(1, 2, 0)
        new_map = new_map.to(self.device)

        new_tex = TexturesUV(
            [new_map], 
            self.mesh.textures.faces_uvs_padded().to(self.device), 
            self.mesh.textures.verts_uvs_padded().to(self.device), 
            sampling_mode=self.sampling_mode
            )
        self.mesh.textures = new_tex


    def set_noise_texture(self, channels=None):
        if not channels:
            channels = self.channels
        noise_texture = torch.normal(0, 1, (channels,) + self.target_size, device=self.device)
        self.set_texture_map(noise_texture)
        return noise_texture


    def set_cameras(self, camera_poses, centers=None, camera_distance=2.0, scale=None):
        elev = torch.FloatTensor([pose[0] for pose in camera_poses])
        azim = torch.FloatTensor([pose[1] for pose in camera_poses])
        
        R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim, at=centers or ((0,0,0),))
        # self.cameras= FoVOrthographicCameras(device=self.device, R=R, T=T) # Either orthographic or perspective
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)


    def set_cameras_and_render_settings(self, camera_poses, centers=None, camera_distance=2.0, render_size=None, scale=None):
        self.set_cameras(camera_poses, centers, camera_distance, scale=scale)
        if render_size is None:
            render_size = self.render_size
        if not hasattr(self, "renderer"):
            self.setup_renderer(size=render_size)
        if not hasattr(self, "mesh_d"):
            self.disconnect_faces()
        if not hasattr(self, "mesh_uv"):
            self.construct_uv_mesh()
        self.calculate_tex_gradient()
        self.calculate_visible_triangle_mask()
        _,_,_,cos_maps,_, _ = self.render_geometry()
        self.calculate_cos_angle_weights(cos_maps)


    def setup_renderer(self, size=64, blur=0.0, face_per_pix=1, perspective_correct=False, channels=None):
        if not channels:
            channels = self.channels
        
        self.raster_settings = RasterizationSettings(
            image_size=size, 
            blur_radius=blur, 
            faces_per_pixel=face_per_pix,
            perspective_correct=perspective_correct,
            cull_backfaces=True,
            max_faces_per_bin=30000,
            bin_size=-1,
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings,

            ),
            shader=HardNChannelFlatShader(
                device=self.device, 
                cameras=self.cameras,
                lights=self.lights,
                channels=channels
            )
        )


    @torch.enable_grad()
    def calculate_cos_angle_weights(self, cos_angles, fill=True, channels=None, disable=None):
        if not channels:
            channels = self.channels
        cos_maps = []
        tmp_mesh = self.mesh.clone()
        for i in range(len(self.cameras)):
            
            zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
            optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
            optimizer.zero_grad()
            zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
            tmp_mesh.textures = zero_tex

            images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights)

            loss = torch.sum((cos_angles[i,:,:,0:1]**1 - images_predicted)**2)
            loss.backward()
            optimizer.step()

            if fill:
                zero_map = zero_map.detach() / (self.gradient_maps[i] + 1E-8)
                zero_map = voronoi_solve(zero_map, self.gradient_maps[i][...,0], disable=disable)
            else:
                zero_map = zero_map.detach() / (self.gradient_maps[i]+1E-8)
            cos_maps.append(zero_map)
        self.cos_maps = cos_maps

        
    @torch.no_grad()
    def render_geometry(self, image_size=None):
        if image_size:
            size = self.renderer.rasterizer.raster_settings.image_size
            self.renderer.rasterizer.raster_settings.image_size = image_size
        shader = self.renderer.shader
        self.renderer.shader = HardGeometryShader(device=self.device, cameras=self.cameras[0], lights=self.lights)
        tmp_mesh = self.mesh.clone()
        
        verts, normals, depths, cos_angles, texels, fragments = self.renderer(tmp_mesh.extend(len(self.cameras)), cameras=self.cameras, lights=self.lights)
        self.renderer.shader = shader

        if image_size:
            self.renderer.rasterizer.raster_settings.image_size = size

        return verts, normals, depths, cos_angles, texels, fragments


    @torch.no_grad()
    def decode_view_normal(self, normals):
        w2v_mat = self.cameras.get_full_projection_transform()
        normals_view = torch.clone(normals)[:,:,:,0:3]
        normals_view = normals_view.reshape(normals_view.shape[0], -1, 3)
        normals_view = w2v_mat.transform_normals(normals_view)
        normals_view = normals_view.reshape(normals.shape[0:3]+(3,))
        normals_view[:,:,:,2] *= -1
        normals = (normals_view[...,0:3]+1) * normals[...,3:] / 2 + torch.FloatTensor(((((0.5,0.5,1))))).to(self.device) * (1 - normals[...,3:])
        normals = normals.clamp(0, 1)
        return normals


    @torch.no_grad()
    def decode_normalized_depth(self, depths, batched_norm=False):
        view_z, mask = depths.unbind(-1)
        view_z = view_z * mask + 100 * (1-mask)
        inv_z = 1 / view_z
        inv_z_min = inv_z * mask + 100 * (1-mask)
        if not batched_norm:
            max_ = torch.max(inv_z, 1, keepdim=True)
            max_ = torch.max(max_[0], 2, keepdim=True)[0]

            min_ = torch.min(inv_z_min, 1, keepdim=True)
            min_ = torch.min(min_[0], 2, keepdim=True)[0]
        else:
            max_ = torch.max(inv_z)
            min_ = torch.min(inv_z_min)
        inv_z = (inv_z - min_) / (max_ - min_)
        inv_z = inv_z.clamp(0,1)
        inv_z = inv_z[...,None].repeat(1,1,1,3)

        return inv_z


    @torch.enable_grad()
    def calculate_tex_gradient(self, channels=None):
        if not channels:
            channels = self.channels
        tmp_mesh = self.mesh.clone()
        gradient_maps = []
        for i in range(len(self.cameras)):
            zero_map = torch.zeros(self.target_size+(channels,), device=self.device, requires_grad=True)
            optimizer = torch.optim.SGD([zero_map], lr=1, momentum=0)
            optimizer.zero_grad()
            zero_tex = TexturesUV([zero_map], self.mesh.textures.faces_uvs_padded(), self.mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
            tmp_mesh.textures = zero_tex
            images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights)
            loss = torch.sum((1 - images_predicted)**2)
            loss.backward()
            optimizer.step()

            gradient_maps.append(zero_map.detach())

        self.gradient_maps = gradient_maps


    @torch.no_grad()
    def calculate_visible_triangle_mask(self, channels=None, image_size=(512,512)):
        if not channels:
            channels = self.channels

        pix2face_list = []
        for i in range(len(self.cameras)):
            self.renderer.rasterizer.raster_settings.image_size=image_size
            pix2face = self.renderer.rasterizer(self.mesh_d, cameras=self.cameras[i]).pix_to_face
            self.renderer.rasterizer.raster_settings.image_size=self.render_size
            pix2face_list.append(pix2face)

        if not hasattr(self, "mesh_uv"):
            self.construct_uv_mesh()

        raster_settings = RasterizationSettings(
            image_size=self.target_size, 
            blur_radius=0, 
            faces_per_pixel=1,
            perspective_correct=False,
            cull_backfaces=False,
            max_faces_per_bin=30000,)

        R, T = look_at_view_transform(dist=2, elev=0, azim=0)
        cameras = FoVOrthographicCameras(device=self.device, R=R, T=T)

        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        uv_pix2face = rasterizer(self.mesh_uv).pix_to_face

        visible_triangles = []
        for i in range(len(pix2face_list)):
            valid_faceid = torch.unique(pix2face_list[i])
            valid_faceid = valid_faceid[1:] if valid_faceid[0]==-1 else valid_faceid
            mask = torch.isin(uv_pix2face[0], valid_faceid, assume_unique=False)
            triangle_mask = torch.ones(self.target_size+(1,), device=self.device)
            triangle_mask[~mask] = 0
            
            triangle_mask[:,1:][triangle_mask[:,:-1] > 0] = 1
            triangle_mask[:,:-1][triangle_mask[:,1:] > 0] = 1
            triangle_mask[1:,:][triangle_mask[:-1,:] > 0] = 1
            triangle_mask[:-1,:][triangle_mask[1:,:] > 0] = 1
            visible_triangles.append(triangle_mask)

        self.visible_triangles = visible_triangles


    def render_textured_views(self):
        meshes = self.mesh.extend(len(self.cameras))
        images_predicted = self.renderer(meshes, cameras=self.cameras, lights=self.lights)

        return [image.permute(2, 0, 1) for image in images_predicted]


    @torch.enable_grad()
    def bake_texture(self, 
                     views=None, 
                     main_views=[], 
                     cos_weighted=True, 
                     channels=None, 
                     exp=None, 
                     noisy=False, 
                     generator=None,
                     disable=None,):

        if not exp:
            exp=1
        if not channels:
            channels = self.channels
        views = [view.permute(1, 2, 0) for view in views] # (H, W, 4)

        tmp_mesh = self.mesh
        bake_maps = [torch.zeros(self.target_size+(views[0].shape[2],), device=self.device, requires_grad=True) for view in views]
        optimizer = torch.optim.SGD(bake_maps, lr=1, momentum=0)
        optimizer.zero_grad()
        loss = 0
        for i in range(len(self.cameras)):    
            bake_tex = TexturesUV([bake_maps[i]], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
            tmp_mesh.textures = bake_tex

            images_predicted = self.renderer(tmp_mesh, cameras=self.cameras[i], lights=self.lights, device=self.device)

            predicted_rgb = images_predicted[..., :-1]

            loss += (((predicted_rgb[...] - views[i]))**2).sum()

        loss.backward(retain_graph=False)
        optimizer.step()

        total_weights = 0
        baked = 0
        for i in range(len(bake_maps)):
            normalized_baked_map = bake_maps[i].detach() / (self.gradient_maps[i] + 1E-8)
            bake_map = voronoi_solve(normalized_baked_map, self.gradient_maps[i][...,0], disable=disable)
            weight = self.visible_triangles[i] * (self.cos_maps[i]) ** exp
            if noisy:
                noise = torch.rand(weight.shape[:-1]+(1,), generator=generator).type(weight.dtype).to(weight.device)
                weight *= noise
            total_weights += weight
            baked += bake_map * weight
        baked /= total_weights + 1E-8
        baked = voronoi_solve(baked, total_weights[...,0], disable=disable)

        bake_tex = TexturesUV([baked], tmp_mesh.textures.faces_uvs_padded(), tmp_mesh.textures.verts_uvs_padded(), sampling_mode=self.sampling_mode)
        tmp_mesh.textures = bake_tex
        extended_mesh = tmp_mesh.extend(len(self.cameras))
        images_predicted = self.renderer(extended_mesh, cameras=self.cameras, lights=self.lights)
        learned_views = [image.permute(2, 0, 1) for image in images_predicted]

        return learned_views, baked.permute(2, 0, 1), total_weights.permute(2, 0, 1)


    def to(self, device):
        for mesh_name in ["mesh", "mesh_d", "mesh_uv"]:
            if hasattr(self, mesh_name):
                mesh = getattr(self, mesh_name)
                setattr(self, mesh_name, mesh.to(device))
        for list_name in ["visible_triangles", "visibility_maps", "cos_maps"]:
            if hasattr(self, list_name):
                map_list = getattr(self, list_name)
                for i in range(len(map_list)):
                    map_list[i] = map_list[i].to(device)
