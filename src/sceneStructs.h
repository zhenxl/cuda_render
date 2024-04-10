#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (3.0f)

enum GeomType {
    SPHERE,
    CUBE,
    TriangleMesh
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct ObjectTransform {
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Object {
    enum GeomType type;
    int materialid;
    int triangleStart, triangleEnd;
    ObjectTransform Transform;
};

struct BoundingBox {
    glm::vec3 pMin, pMax;
    BoundingBox() : pMin(glm::vec3(1e38f)), pMax(glm::vec3(-1e38f)) {}
    glm::vec3 center() const { return (pMin + pMax) * 0.5f; }
};

BoundingBox Union(const BoundingBox& b1, const BoundingBox& b2);
BoundingBox Union(const BoundingBox& b1, const glm::vec3& p);
float BoxArea(const BoundingBox& b);

struct Primitive {
    int objID;
    int offset;
    BoundingBox bbox;
    Primitive(const Object& obj, int objID, int triangleOffset = -1, const glm::ivec3* triangles = nullptr, const glm::vec3* vertices = nullptr);
};

struct BVHNode {
    int axis;
    BVHNode* left, * right;
    int startPrim, endPrim;
    BoundingBox bbox;
};

struct BVHGPUNode {
    int axis;
    BoundingBox bbox;
    int parent, left, right;
    int startPrim, endPrim;
    BVHGPUNode() :axis(-1), parent(-1), left(-1), right(-1), startPrim(-1), endPrim(-1) {}
};

enum MaterialType {
    diffuse, frenselSpecular, emitting, microfacet, metallicWorkflow, blinnphong
};
struct Material {
    glm::vec3 color = glm::vec3(0);
    float indexOfRefraction = 0;
    float emittance = 0;
    float roughness = -1.0;
    float metallic = -1.0;
    float specExponent = -1.0;
    cudaTextureObject_t baseColorMap = 0, normalMap = 0, metallicRoughnessMap = 0;
    MaterialType type = diffuse;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lensRadius = 0.001f;
    float focalLength = 1.0f;
};

struct ModelInfoDevice {
    glm::ivec3* dev_triangles;
    glm::vec3* dev_vertices;
    glm::vec2* dev_uvs;
    glm::vec3* dev_normals;
};

struct  SceneInfoDevice {
    Material* dev_materials;
    Object* dev_objs;
    int objectsSize;
    ModelInfoDevice modelInfo;
    Primitive* dev_primitives;
    BVHGPUNode* dev_bvhGPUArray;
    int bvhDataSize;
    Primitive* dev_lights;
    int lightsSize;
    cudaTextureObject_t skyboxObj;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 transport;
    int pixelIndex;
    int remainingBounces;
    float lastPdf;
};

enum TextureType {
    color, normal, metallicroughness
};

struct GLTFTextureLoadInfo {
    char* buffer;
    int matIndex;
    TextureType texType;
    int width, height;
    int bits, component;
    GLTFTextureLoadInfo(char* buffer, int index, TextureType type, int width, int height, int bits, int component) :buffer(buffer), matIndex(index), texType(type), width(width), height(height), bits(bits), component(component) {}
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t = -1.0;
  glm::vec3 surfaceNormal = glm::vec3(0.0);
  glm::vec3 worldPos = glm::vec3(0.0);
  int materialId = -1;
  int primitiveId = -1;
  glm::vec2 uv = glm::vec2(0.0);
  MaterialType type = diffuse;
};
