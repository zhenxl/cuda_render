#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define SCATTER_ORIGIN_OFFSETMULT 0.001f

#define MAX_ITER 8
#define USE_BVH 1
#define TONEMAPPING 1
#define DOF_ENABLED 1
#define STOCHASTIC_SAMPLING 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

__device__ inline bool util_math_is_nan(const glm::vec3& v)
{
	return (v.x != v.x) || (v.y != v.y) || (v.z != v.z);
}

__device__ inline glm::vec2 util_concentric_sample_disk(glm::vec2 rand)
{
	rand = 2.0f * rand - 1.0f;
	if (rand.x == 0 && rand.y == 0)
	{
		return glm::vec2(0);
	}
	const float pi_4 = PI / 4, pi_2 = PI / 2;
	bool x_g_y = abs(rand.x) > abs(rand.y);
	float theta = x_g_y ? pi_4 * rand.y / rand.x : pi_2 - pi_4 * rand.x / rand.y;
	float r = x_g_y ? rand.x : rand.y;
	return glm::vec2(cos(theta), sin(theta)) * r;
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::vec3 color;
#if TONEMAPPING
		color = pix / (float)iter;
		color = util_postprocess_gamma(util_postprocess_ACESFilm(color));
		color = color * 255.0f;
#else
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);
#endif

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		if (util_math_is_nan(color)) {
			pbo[index].x = 255;
			pbo[index].y = 192;
			pbo[index].z = 203;
		}
		else {
			pbo[index].x = color.x;
			pbo[index].y = color.y;
			pbo[index].z = color.z;
		}
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Object* dev_geoms = NULL;
static Material* dev_materials = NULL;
static glm::ivec3* dev_triangles = NULL;
static glm::vec3* dev_vertices = NULL;
static glm::vec2* dev_uvs = NULL;
static glm::vec3* dev_normals = NULL;
static BVHGPUNode* dev_bvhArray = NULL;
static Primitive* dev_primitives = NULL;
static Primitive* dev_lights = NULL;
static PathSegment* dev_paths1 = NULL;
static PathSegment* dev_paths2 = NULL;
static ShadeableIntersection* dev_intersections1 = NULL;
static ShadeableIntersection* dev_intersections2 = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths1, pixelcount * sizeof(PathSegment));
	cudaMalloc(&dev_paths2, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->objects.size() * sizeof(Object));
	cudaMemcpy(dev_geoms, scene->objects.data(), scene->objects.size() * sizeof(Object), cudaMemcpyHostToDevice);

	if (scene->triangles.size())
	{
		cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(glm::ivec3));
		cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(glm::ivec3), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_vertices, scene->verticies.size() * sizeof(glm::vec3));
		cudaMemcpy(dev_vertices, scene->verticies.data(), scene->verticies.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_uvs, scene->uvs.size() * sizeof(glm::vec2));
		cudaMemcpy(dev_uvs, scene->uvs.data(), scene->uvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);

		if (scene->normals.size())
		{
			cudaMalloc(&dev_normals, scene->normals.size() * sizeof(glm::vec3));
			cudaMemcpy(dev_normals, scene->normals.data(), scene->normals.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		}
	}

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections1, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_bvhArray, scene->bvhArray.size() * sizeof(BVHGPUNode));
	cudaMemcpy(dev_bvhArray, scene->bvhArray.data(), scene->bvhArray.size() * sizeof(BVHGPUNode), cudaMemcpyHostToDevice);
	///*printf("in pathtrace init");*/
	//printf("bvhArray size %d\n", scene->bvhArray.size());
	//printf("dev_bvhArray: %d\n", scene->bvhArray[1].left);
	///*if (dev_bvhArray[0].left > -10) {
	//	
	//}*/
	//printf("finished");


	cudaMalloc(&dev_primitives, scene->primitives.size() * sizeof(Primitive));
	cudaMemcpy(dev_primitives, scene->primitives.data(), scene->primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice);

	if (scene->lights.size()) {
		cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Primitive));
		cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Primitive), cudaMemcpyHostToDevice);
	}
	

	cudaMalloc(&dev_intersections2, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections2, 0, pixelcount * sizeof(ShadeableIntersection));

	// TODO: initialize any extra device memeory you need

	checkCUDAError("pathtraceInit");
}

void pathtraceFree(Scene * scene) {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths1);
	cudaFree(dev_paths2);
	cudaFree(dev_geoms);
	if (scene->triangles.size()) {
		cudaFree(dev_triangles);
		cudaFree(dev_vertices);
	}
	cudaFree(dev_materials);
	cudaFree(dev_intersections1);
	cudaFree(dev_intersections2);
	// TODO: clean up any extra device memory you created

	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	thrust::default_random_engine rng = makeSeededRandomEngine(x, y, iter);
	thrust::uniform_real_distribution<float> u01(0, 1);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.transport = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
#if STOCHASTIC_SAMPLING
		// TODO: implement antialiasing by jittering the ray
		glm::vec2 jitter = glm::vec2(0.5f * (u01(rng) * 2.0f - 1.0f), 0.5f * (u01(rng) * 2.0f - 1.0f));
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter[0])
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter[1])
		);
#if DOF_ENABLED
		float lensR = cam.lensRadius;
		glm::vec3 perpDir = glm::cross(cam.right, cam.up);
		perpDir = glm::normalize(perpDir);
		float focalLen = cam.focalLength;
		float tFocus = focalLen / glm::abs(glm::dot(segment.ray.direction, perpDir));
		glm::vec2 offset = lensR * util_concentric_sample_disk(glm::vec2(u01(rng), u01(rng)));
		glm::vec3 newOri = offset.x * cam.right + offset.y * cam.up + cam.position;
		glm::vec3 pFocus = segment.ray.direction * tFocus + segment.ray.origin;
		segment.ray.direction = glm::normalize(pFocus - newOri);
		segment.ray.origin = newOri;
#endif

#else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		segment.lastPdf = -1.0;
	}
}


__global__ void compute_intersection_with_bvh(
	int num_paths
	, PathSegment* pathSegments
	, const SceneInfoDevice sceneInfo
	, ShadeableIntersection* intersections
	, int* rayValid
	, glm::vec3* image
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) return;
	PathSegment& pathSegment = pathSegments[path_index];
	glm::vec3 ray_dir = pathSegment.ray.direction;
	glm::vec3 ray_ori = pathSegment.ray.origin;
	int curr = bvh_get_near_child(sceneInfo.dev_bvhGPUArray, 0, ray_ori);
	ShadeableIntersection curr_intersect;
	curr_intersect.t = 1e20f;
	bool intersected = false;
	Bvh_traverse_state state = fromParent;
	//printf("curr %d\n", curr);
	/*return;*/
	while (curr >= 0 && curr < sceneInfo.bvhDataSize) {
		//printf("in loop");
		if (state == fromChild) {
			if (curr == 0) break;
			int parent = sceneInfo.dev_bvhGPUArray[curr].parent;
			if (curr == bvh_get_near_child(sceneInfo.dev_bvhGPUArray, parent, pathSegment.ray.origin)) {
				curr = bvh_get_sibling(sceneInfo.dev_bvhGPUArray, curr);
				state = fromSibling;
			}
			else {
				curr = parent;
				state = fromChild;
			}
		}
		else if (state == fromSibling) {
			if (!boundingBoxIntersectionTest(sceneInfo.dev_bvhGPUArray[curr].bbox, pathSegment.ray)) {
				curr = sceneInfo.dev_bvhGPUArray[curr].parent;
				state = fromChild;
			}
			else if (bvh_is_leaf(sceneInfo.dev_bvhGPUArray, curr)) {
				//printf("bvh has leaf in fromSibling");
				if (bvh_leaf_intersect(curr, pathSegment.ray, sceneInfo, &curr_intersect)) {
					intersected = true;

					//printf("now intersected \n");
					//break;
				}
				curr = sceneInfo.dev_bvhGPUArray[curr].parent;
				state = fromChild;
			}
			else {
				curr = bvh_get_near_child(sceneInfo.dev_bvhGPUArray, curr, pathSegment.ray.origin);
				state = fromParent;
			}

		} else { //fromParent
			if (!boundingBoxIntersectionTest(sceneInfo.dev_bvhGPUArray[curr].bbox, pathSegment.ray)) {
				curr = bvh_get_sibling(sceneInfo.dev_bvhGPUArray, curr);
				state = fromSibling;
			}
			else if (bvh_is_leaf(sceneInfo.dev_bvhGPUArray, curr)) {
				//printf("bvh has leaf in fromParent");
				if (bvh_leaf_intersect(curr, pathSegment.ray, sceneInfo, &curr_intersect)) {
					intersected = true;
					//printf("now intersected \n");
					//break;
				}
				curr = bvh_get_sibling(sceneInfo.dev_bvhGPUArray, curr);
				state = fromSibling;
			}
			else {
				curr = bvh_get_near_child(sceneInfo.dev_bvhGPUArray, curr, pathSegment.ray.origin);
				state = fromParent;
			}
		}
	}

	rayValid[path_index] = intersected ? 1:0;
	if (intersected) {
		//printf("has intersected");
		intersections[path_index]      = curr_intersect;
		intersections[path_index].type = sceneInfo.dev_materials[curr_intersect.materialId].type;
	}
	else if (sceneInfo.skyboxObj)
	{ 
		//printf("has in skyboxTex\n");
		glm::vec2 uv = util_sample_spherical_map(glm::normalize(ray_dir));
		float4 skyColorRGBA = tex2D<float4>(sceneInfo.skyboxObj, uv.x, uv.y);
		glm::vec3 skyColor = glm::vec3(skyColorRGBA.x, skyColorRGBA.y, skyColorRGBA.z);
		image[pathSegment.pixelIndex] += pathSegment.transport * skyColor * BACKGROUND_COLOR;
	}
	//printf("skyboxTex %d\n", skyboxTex);

}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Object* geoms
	, int geoms_size
	, glm::ivec3* modelTriangles
	, glm::vec3* modelVertices
	, ShadeableIntersection* intersections
	, int *ray_valid
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment& pathSegment = pathSegments[path_index];
		float t = -1.0;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_material_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec3 baryCoord;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Object& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == TriangleMesh)
			{
				//printf("in model \n");
				glm::vec3 baryCoord;
				for (int i = geom.triangleStart; i != geom.triangleEnd; i++)
				{
					const glm::ivec3& tri = modelTriangles[i];
					const glm::vec3& v0 = modelVertices[tri[0]];
					const glm::vec3& v1 = modelVertices[tri[1]];
					const glm::vec3& v2 = modelVertices[tri[2]];
					t = triangleIntersectionTest(geom.Transform, v0, v1, v2, pathSegment.ray, tmp_intersect, tmp_normal, baryCoord);
					if (t > 0.0f && t_min > t)
					{
						//printf("intersected \n");
						t_min = t;
						hit_material_index = geom.materialid;
						intersect_point = tmp_intersect;
						normal = tmp_normal;
					}
				}
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_material_index = geom.materialid;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (t_min == FLT_MAX)//hits nothing
		{
			ray_valid[path_index] = 0;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = hit_material_index;
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].worldPos = intersect_point;
			ray_valid[path_index] = 1;
		}
	}
}

__global__ void scatter_on_intersection_mis(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, SceneInfoDevice sceneInfo
	, PathSegment* pathSegments
	, Material* materials
	, int* rayValid
	, glm::vec3* image) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		//printf("before light sample %d\n", sceneInfo.lightsSize);
		ShadeableIntersection intersection = shadeableIntersections[idx];
		// Set up the RNG
		// LOOK: this is how you use thrust's RNG! Please look at
		// makeSeededRandomEngine as well.
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		// If the material indicates that the object was a light, "light" the ray
		if (material.type == MaterialType::emitting) {
			float matPdf = pathSegments[idx].lastPdf;
			int lightPrimId = intersection.primitiveId;
			if (matPdf > 0.0) {
				float G = util_math_solid_angle_to_area(intersection.worldPos, intersection.surfaceNormal, pathSegments[idx].ray.origin);
				//We do not know the value of light pdf(of last intersection point) of the sample taken from bsdf sampling unless we hit a light
				float lightPdf = lights_sample_pdf(sceneInfo, lightPrimId);
				//Computing weights from last intersection point
				float misW = util_mis_weight(matPdf * G, lightPdf);
				pathSegments[idx].transport *= (materialColor * material.emittance * misW);
			}
			else {
				pathSegments[idx].transport *= (materialColor * material.emittance);
			}
			rayValid[idx] = 0;
			if (!util_math_is_nan(pathSegments[idx].transport))
				image[pathSegments[idx].pixelIndex] += pathSegments[idx].transport;

			//pathSegments[idx].lastPdf = 0.0;
		}
		
		else {
			glm::vec3& woInWorld = pathSegments[idx].ray.direction;
			glm::vec3& N = glm::normalize(intersection.surfaceNormal);
			glm::vec3 B, T;
			util_math_get_TBN_pixar(N, &T, &B);
			glm::mat3 TBN(T, B, N);
			glm::vec3 wo = glm::transpose(TBN) * (-woInWorld);
			wo = glm::normalize(wo);
			float pdf = 0;
			float cosWi = 0.0;
			glm::vec3 wi, bxdf;
			
			if (material.type == MaterialType::frenselSpecular) {
				glm::vec2 iors = glm::dot(woInWorld, N) < 0 ? glm::vec2(1.0, material.indexOfRefraction) : glm::vec2(material.indexOfRefraction, 1.0);
				bxdf = bxdf_frensel_specular_sample_f(wo, &wi, glm::vec2(u01(rng), u01(rng)), &pdf, materialColor, materialColor, iors);
				cosWi = 1.0;
			}
			else {
				glm::vec3 light_bxdf = glm::vec3(0);
				float roughness = material.roughness, metallic = material.metallic;

				glm::vec3 lightPos, lightNormal, emissive = glm::vec3(0);
				float light_pdf = -1.0;
				glm::vec3 offseted_pos = intersection.worldPos + N * SCATTER_ORIGIN_OFFSETMULT;
				//printf("before light sample %d\n", sceneInfo.bvhDataSize);
				lights_sample(sceneInfo, glm::vec3(u01(rng), u01(rng), u01(rng)), offseted_pos, N, &lightPos, &lightNormal, &emissive, &light_pdf);
				//printf("after light sample\n");


				if (emissive.x > 0.0 || emissive.y > 0.0 || emissive.z > 0.0) {
					glm::vec3 light_wi = lightPos - offseted_pos;
					light_wi = glm::normalize(glm::transpose(TBN) * (light_wi));
					float G = util_math_solid_angle_to_area(lightPos, lightNormal, offseted_pos);
					float mat_pdf = -1.0f;

					if (material.type == MaterialType::diffuse) {
						mat_pdf = bxdf_diffuse_pdf(wo, light_wi);
						light_bxdf = bxdf_diffuse_eval(wo, light_wi, materialColor);
					}
					else if (material.type == MaterialType::microfacet) {
						mat_pdf = bxdf_microfacet_pdf(wo, light_wi, roughness);
						light_bxdf = bxdf_microfacet_eval(wo, light_wi, materialColor, roughness);
					}
					else if (material.type == MaterialType::metallicWorkflow) {
						mat_pdf = bxdf_metallic_workflow_pdf(wo, light_wi, materialColor, metallic, roughness);
						light_bxdf = bxdf_metallic_workflow_eval(wo, light_wi, materialColor, metallic, roughness);
					}
					float misW = util_mis_weight(light_pdf, mat_pdf * G);
					image[pathSegments[idx].pixelIndex] += pathSegments[idx].transport * light_bxdf * util_math_tangent_space_clampedcos(light_wi) * emissive * misW * G / light_pdf;
				}

				if (material.type == MaterialType::diffuse) {
					float4 color = { 0,0,0,1 };
					if (material.baseColorMap != 0) {
						color = tex2D<float4>(material.baseColorMap, intersection.uv.x, intersection.uv.y);
						materialColor.x = color.x;
						materialColor.y = color.y;
						materialColor.z = color.z;
					}
					bxdf = bxdf_diffuse_sample_f(wo, &wi, glm::vec2(u01(rng), u01(rng)), &pdf, materialColor);
				}
				else if (material.type == MaterialType::microfacet) {
					bxdf = bxdf_microfacet_sample_f(wo, &wi, glm::vec2(u01(rng), u01(rng)), &pdf, materialColor, material.roughness);
				}
				else if (material.type == MaterialType::metallicWorkflow) {
					bxdf = bxdf_metallic_workflow_sample_f(wo, &wi, glm::vec3(u01(rng), u01(rng), u01(rng)), &pdf, materialColor, metallic, roughness);
				}

				cosWi = util_math_tangent_space_clampedcos(wi);
			}
			if (pdf > 0)
			{
				pathSegments[idx].transport *= bxdf * cosWi / pdf;
				glm::vec3 newDir = glm::normalize(TBN * wi);
				glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
				float offsetMult = material.type != MaterialType::frenselSpecular ? SCATTER_ORIGIN_OFFSETMULT : SCATTER_ORIGIN_OFFSETMULT * 100.0f;
				pathSegments[idx].ray.origin = intersection.worldPos + offset * offsetMult;
				pathSegments[idx].ray.direction = newDir;
				pathSegments[idx].lastPdf = pdf;
				rayValid[idx] = 1;
			}
			else
			{
				rayValid[idx] = 0;
			}
		}
	}
}

__global__ void scatter_on_intersection(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, int* rayValid
	, glm::vec3* image
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		// Set up the RNG
		// LOOK: this is how you use thrust's RNG! Please look at
		// makeSeededRandomEngine as well.
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		glm::vec3 materialColor = material.color;

		// If the material indicates that the object was a light, "light" the ray
		if (material.type ==  MaterialType::emitting) {
			pathSegments[idx].transport *= (materialColor * material.emittance);
			rayValid[idx] = 0;
			if (!util_math_is_nan(pathSegments[idx].transport))
				image[pathSegments[idx].pixelIndex] += pathSegments[idx].transport;
		}
		else {
			glm::vec3& woInWorld = pathSegments[idx].ray.direction;
			glm::vec3& N = glm::normalize(intersection.surfaceNormal);
			glm::vec3 B, T;
			util_math_get_TBN_pixar(N, &T, &B);
			glm::mat3 TBN(T, B, N);
			glm::vec3 wo = glm::transpose(TBN) * (-woInWorld);
			wo = glm::normalize(wo);
			float pdf = 0;
			glm::vec3 wi, bxdf;
			if (material.type == MaterialType::diffuse) {
				float4 color = { 0,0,0,1 };
				if (material.baseColorMap != 0) {
					color = tex2D<float4>(material.baseColorMap, intersection.uv.x, intersection.uv.y);
					materialColor.x = color.x;
					materialColor.y = color.y;
					materialColor.z = color.z;
				}
				bxdf = bxdf_diffuse_sample_f(wo, &wi, glm::vec2(u01(rng), u01(rng)), &pdf, materialColor);
			}
			else if (material.type == MaterialType::frenselSpecular) {
				glm::vec2 iors = glm::dot(woInWorld, N) < 0 ? glm::vec2(1.0, material.indexOfRefraction) : glm::vec2(material.indexOfRefraction, 1.0);
				bxdf = bxdf_frensel_specular_sample_f(wo, &wi, glm::vec2(u01(rng), u01(rng)), &pdf, materialColor, materialColor, iors);
			}
			else if (material.type == MaterialType::microfacet) {
				bxdf = bxdf_microfacet_sample_f(wo, &wi, glm::vec2(u01(rng), u01(rng)), &pdf, materialColor, material.roughness);
			}
			if (pdf > 0)
			{
				if (material.type == MaterialType::frenselSpecular) {
					pathSegments[idx].transport *= 1.f *  bxdf /pdf;
				}
				else {
					pathSegments[idx].transport *= bxdf * util_math_tangent_space_clampedcos(wi) / pdf;
				}
				glm::vec3 newDir = glm::normalize(TBN * wi);
				glm::vec3 offset = glm::dot(newDir, N) < 0 ? -N : N;
				float offsetMult = material.type != MaterialType::frenselSpecular ? SCATTER_ORIGIN_OFFSETMULT : SCATTER_ORIGIN_OFFSETMULT * 100.0f;
				pathSegments[idx].ray.origin = intersection.worldPos + offset * offsetMult;
				pathSegments[idx].ray.direction = newDir;
				rayValid[idx] = 1;
			}
			else
			{
				rayValid[idx] = 0;
			}
		}
	}
}


struct mat_comp {
	__host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
		return a.type < b.type;
	}
};


int compact_rays(int *rayValid, int *rayIndex, int numRays, bool sortByMat=false) {
	thrust::device_ptr<PathSegment> dev_thrust_path1(dev_paths1), dev_thrust_path2(dev_paths2);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersect1(dev_intersections1), dev_thrust_intersect2(dev_intersections2);
	thrust::device_ptr<int> dev_thrust_rayValid(rayValid), dev_thrust_rayIndex(rayIndex);
	thrust::exclusive_scan(dev_thrust_rayValid, dev_thrust_rayValid + numRays, dev_thrust_rayIndex);
	//printf("numRays: %d in compact rays\n", numRays);
	int next_numRays, tmp;
	cudaMemcpy(&tmp, rayIndex + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	next_numRays = tmp;
	cudaMemcpy(&tmp, rayValid + numRays - 1, sizeof(int), cudaMemcpyDeviceToHost);
	next_numRays += tmp;
	thrust::scatter_if(dev_thrust_path1, dev_thrust_path1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_path2);
	thrust::scatter_if(dev_thrust_intersect1, dev_thrust_intersect1 + numRays, dev_thrust_rayIndex, dev_thrust_rayValid, dev_thrust_intersect2);
	if (sortByMat) {
		mat_comp cmp;
		thrust::sort_by_key(dev_thrust_intersect2, dev_thrust_intersect2 + next_numRays, dev_thrust_path2, cmp);
	}
	std::swap(dev_paths1, dev_paths2);
	std::swap(dev_intersections1, dev_intersections2);
	return next_numRays;
}



// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		if (!util_math_is_nan(iterationPath.transport))
			image[iterationPath.pixelIndex] += iterationPath.transport;
	}
}



/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
 /**
  * Wrapper for the __global__ call that sets up the kernel calls and does a ton
  * of memory management
  */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing
	SceneInfoDevice dev_sceneInfo{};
	dev_sceneInfo.dev_materials           = dev_materials;
	dev_sceneInfo.dev_objs                = dev_geoms;
	dev_sceneInfo.objectsSize             = hst_scene->objects.size();
	dev_sceneInfo.dev_materials           = dev_materials;
	dev_sceneInfo.dev_bvhGPUArray         = dev_bvhArray;
	dev_sceneInfo.bvhDataSize             = hst_scene->bvhArray.size();
	dev_sceneInfo.dev_primitives          = dev_primitives;
	dev_sceneInfo.skyboxObj               = hst_scene->skyboxTextureObj;
	dev_sceneInfo.dev_lights              = dev_lights;
	dev_sceneInfo.lightsSize              = hst_scene->lights.size();
	//printf("light size: %d\n", dev_sceneInfo.lightsSize);
	dev_sceneInfo.modelInfo.dev_triangles = dev_triangles;
	dev_sceneInfo.modelInfo.dev_vertices  = dev_vertices;
	dev_sceneInfo.modelInfo.dev_normals   = dev_normals;
	dev_sceneInfo.modelInfo.dev_uvs       = dev_uvs;


	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>> (cam, iter, traceDepth, dev_paths1);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths1 + pixelcount;
	int num_paths = dev_path_end - dev_paths1;
	int* rayValid, * rayIndex;

	int numRays = num_paths;
	cudaMalloc((void**)&rayValid, sizeof(int) * pixelcount);
	cudaMalloc((void**)&rayIndex, sizeof(int) * pixelcount);
	//printf("pixel count: %d\n", pixelcount);
	cudaDeviceSynchronize();
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (numRays && depth < MAX_ITER) {

		// clean shading chunks
		cudaMemset(dev_intersections1, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (numRays + blockSize1d - 1) / blockSize1d;
#if USE_BVH
		compute_intersection_with_bvh <<<numblocksPathSegmentTracing, blockSize1d >>> (
			numRays
			, dev_paths1
			, dev_sceneInfo
			, dev_intersections1
			, rayValid
			, dev_image
			);
#else
		computeIntersections <<<numblocksPathSegmentTracing, blockSize1d >>> (
			depth
			, numRays
			, dev_paths1
			, dev_geoms
			, hst_scene->objects.size()
			, dev_triangles
			, dev_vertices
			, dev_intersections1
			, rayValid
			);
#endif
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

		//printf("first compact rays\n");

		numRays = compact_rays(rayValid, rayIndex, numRays);

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.
		if (!numRays) break;
		dim3 numblocksLightScatter = (numRays + blockSize1d - 1) / blockSize1d;
		//scatter_on_intersection << <numblocksPathSegmentTracing, blockSize1d >> > (
		//	iter,
		//	numRays,
		//	dev_intersections1,
		//	//dev_sceneInfo,
		//	dev_paths1,
		//	dev_materials,
		//	rayValid,
		//	dev_image
		//	);
		//printf("light size before mis: %d\n", dev_sceneInfo.lightsSize);
		scatter_on_intersection_mis << <numblocksLightScatter, blockSize1d >> > (
			iter,
			numRays,
			dev_intersections1,
			dev_sceneInfo,
			dev_paths1,
			dev_materials,
			rayValid,
			dev_image
			);
		//printf("numRays: %d\n", numRays);
		numRays = compact_rays(rayValid, rayIndex, numRays);

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}

	if (numRays)
	{
		// Assemble this iteration and apply it to the image
		dim3 numBlocksPixels = (numRays + blockSize1d - 1) / blockSize1d;
		finalGather << <numBlocksPixels, blockSize1d >> > (numRays, dev_image, dev_paths1);
	}

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	cudaFree(rayValid);
	cudaFree(rayIndex);

	checkCUDAError("pathtrace");
}