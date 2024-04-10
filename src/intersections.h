#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__host__ __device__ inline float util_geometry_ray_box_intersection(const glm::vec3& pMin, const glm::vec3& pMax, const Ray& r, glm::vec3* normal = nullptr)
{
    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = r.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (pMin[xyz] - r.origin[xyz]) / qdxyz;
            float t2 = (pMax[xyz] - r.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax > tmin && tmax > 0) {
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
        }
        if (normal)
            (*normal) = glm::normalize(tmin_n);
        return tmin;
    }
    return -1;
}

__host__ __device__ bool boundingBoxIntersectionTest(const BoundingBox& bbox, const Ray& r)
{
    return util_geometry_ray_box_intersection(bbox.pMin, bbox.pMax, r) > 0.0;
}



// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Object box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.Transform.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.Transform.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.Transform.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.Transform.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Object sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.Transform.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.Transform.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.Transform.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.Transform.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool rayTriangleIntersect(
    const glm::vec3& orig, const glm::vec3& dir,
    const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
    float& t, glm::vec3& normal, glm::vec3& baryCoord)
{
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    glm::vec3 N = glm::cross(v0v1, v0v2); // N
    float area2 = N.length();

    float NdotRayDirection = glm::dot(N, dir);
    if (abs(NdotRayDirection) < 1e-10f) // almost 0
        return false; // they are parallel, so they don't intersect! 

    float d = glm::dot(-N, v0);

    t = -(glm::dot(N, orig) + d) / NdotRayDirection;

    if (t < 0) return false;
    glm::vec3 P = orig + t * dir;

    // Step 2: inside-outside test
    glm::vec3 C;

    // edge 0
    glm::vec3 edge0 = v1 - v0;
    glm::vec3 vp0 = P - v0;
    C = glm::cross(edge0, vp0);
    if (glm::dot(N, C) < 0) return false;
    float areaV2 = glm::length(C);

    // edge 1
    glm::vec3 edge1 = v2 - v1;
    glm::vec3 vp1 = P - v1;
    C = glm::cross(edge1, vp1);
    if (glm::dot(N, C) < 0)  return false;
    float areaV0 = glm::length(C);

    // edge 2
    glm::vec3 edge2 = v0 - v2;
    glm::vec3 vp2 = P - v2;
    C = glm::cross(edge2, vp2);
    if (glm::dot(N, C) < 0) return false;
    float areaV1 = glm::length(C);

    float area = glm::length(N);

    N /= area;
    normal = N;
    baryCoord[0] = areaV0 / area;
    baryCoord[1] = areaV1 / area;
    baryCoord[2] = areaV2 / area;

    return true;
}

__host__ __device__ float triangleIntersectionTest(const ObjectTransform& Transform, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const Ray& r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec3& barycoord)
{
    float t = -1.0;
    glm::vec3 v0w = multiplyMV(Transform.transform, glm::vec4(v0, 1.0f));
    glm::vec3 v1w = multiplyMV(Transform.transform, glm::vec4(v1, 1.0f));
    glm::vec3 v2w = multiplyMV(Transform.transform, glm::vec4(v2, 1.0f));
    if (rayTriangleIntersect(r.origin, r.direction, v0w, v1w, v2w, t, normal, barycoord))
    {
        intersectionPoint = r.origin + r.direction * t;
        return t;
    }
    else
    {
        return -1;
    }
}


enum Bvh_traverse_state {
    fromChild, fromParent, fromSibling
};

__device__ inline int bvh_get_near_child(const BVHGPUNode* bvhArray, int curr, const glm::vec3& ray_ori) {
    //printf("curr %d left %d right %d\n", curr, bvhArray[curr].left, bvhArray[curr].right);
    glm::vec3 left_center = (bvhArray[bvhArray[curr].left].bbox.pMax + bvhArray[bvhArray[curr].left].bbox.pMin) * 0.5f;
    glm::vec3 right_center = (bvhArray[bvhArray[curr].right].bbox.pMax + bvhArray[bvhArray[curr].right].bbox.pMin) * 0.5f;
    float dist_to_left = glm::distance(left_center, ray_ori);
    float dist_to_right = glm::distance(right_center, ray_ori);
    return dist_to_left < dist_to_right ? bvhArray[curr].left : bvhArray[curr].right;
}

__device__ inline int bvh_get_sibling(const BVHGPUNode* bvhArray, int curr) {
    int parent = bvhArray[curr].parent;
    if (parent == -1) return -1;
    return bvhArray[parent].left == curr ? bvhArray[parent].right : bvhArray[parent].left;
}

__device__ inline bool bvh_is_leaf(const BVHGPUNode* bvhArray, int curr) {
    //printf("now bvh is leaf\n");
    return bvhArray[curr].left == -1 && bvhArray[curr].right == -1;
}

__device__ inline bool bvh_leaf_intersect(int curr, const Ray& ray, const SceneInfoDevice& sceneInfo, ShadeableIntersection* intersection) {
    glm::vec3 curr_intersect, curr_normal, curr_barycoord;
    glm::vec2 cur_uv;
    float t;
    bool intersected = false;
    bool outside;
    int primStart = sceneInfo.dev_bvhGPUArray[curr].startPrim, primEnd = sceneInfo.dev_bvhGPUArray[curr].endPrim;
    for (int i = primStart; i < primEnd; i++) {
        const Primitive prim = sceneInfo.dev_primitives[i];
        int objID = prim.objID;
        const Object& obj = sceneInfo.dev_objs[objID];
        if (obj.type == TriangleMesh) {
            const glm::ivec3 tri = sceneInfo.modelInfo.dev_triangles[obj.triangleStart + prim.offset];
            const glm::vec3 v0 = sceneInfo.modelInfo.dev_vertices[tri[0]];
            const glm::vec3 v1 = sceneInfo.modelInfo.dev_vertices[tri[1]];
            const glm::vec3 v2 = sceneInfo.modelInfo.dev_vertices[tri[2]];
            t = triangleIntersectionTest(obj.Transform, v0, v1, v2, ray, curr_intersect, curr_normal, curr_barycoord);

            if (sceneInfo.modelInfo.dev_uvs)
            {
                const glm::vec2& uv0 = sceneInfo.modelInfo.dev_uvs[tri[0]];
                const glm::vec2& uv1 = sceneInfo.modelInfo.dev_uvs[tri[1]];
                const glm::vec2& uv2 = sceneInfo.modelInfo.dev_uvs[tri[2]];
                cur_uv = uv0 * curr_barycoord[0] + uv1 * curr_barycoord[1] + uv2 * curr_barycoord[2];
            }
            if (sceneInfo.modelInfo.dev_normals)
            {
                const glm::vec3& n0 = sceneInfo.modelInfo.dev_normals[tri[0]];
                const glm::vec3& n1 = sceneInfo.modelInfo.dev_normals[tri[1]];
                const glm::vec3& n2 = sceneInfo.modelInfo.dev_normals[tri[2]];
                curr_normal = n0 * curr_barycoord[0] + n1 * curr_barycoord[1] + n2 * curr_barycoord[2];
                curr_normal = glm::vec3(obj.Transform.invTranspose * glm::vec4(curr_normal, 0.0));//TODO: precompute transformation
            }
        }
        else if (obj.type == SPHERE) {
            t = sphereIntersectionTest(obj, ray, curr_intersect, curr_normal, outside);
        }
        else if (obj.type == CUBE) {
            t = boxIntersectionTest(obj, ray, curr_intersect, curr_normal, outside);
        }
        if (t > 0.0 && t < intersection->t) {
            intersection->t = t;
            intersection->materialId = obj.materialid;
            intersection->worldPos = curr_intersect;
            intersection->surfaceNormal = curr_normal;
            intersection->uv = cur_uv;
            intersection->primitiveId = i;
            intersected = true;
            //printf("has intersected in bvh leaf intesect");
        }
    }
    return intersected;
}

__device__ inline float util_math_solid_angle_to_area(glm::vec3 surfacePos, glm::vec3 surfaceNormal, glm::vec3 receivePos)
{
    glm::vec3 L = receivePos - surfacePos;
    return glm::abs(glm::dot(glm::normalize(L), surfaceNormal)) / glm::distance2(surfacePos, receivePos);
}

__host__ __device__ bool util_test_visibility(glm::vec3 p0, glm::vec3 p1, const SceneInfoDevice& dev_sceneInfo)
{
    glm::vec3 dir = p1 - p0;
    bool outside;
    if (glm::length(dir) < 0.001f) return true;
    Ray ray;
    ray.direction = glm::normalize(dir);
    ray.origin = p0;
    glm::vec3 t3 = (dir / ray.direction);
    float t, tmax = max(t3.x, max(t3.y, t3.z)) - 0.001f;
    glm::vec3 tmp_intersect, tmp_normal, tmp_baryCoord;
    for (int i = 0; i < dev_sceneInfo.objectsSize; i++)
    {
        Object& obj = dev_sceneInfo.dev_objs[i];
        if (obj.type == GeomType::CUBE)
        {
            t = boxIntersectionTest(obj, ray, tmp_intersect, tmp_normal, outside);
            if (t > 0.0 && t < tmax) return false;
        }
        else if (obj.type == GeomType::SPHERE)
        {
            t = sphereIntersectionTest(obj, ray, tmp_intersect, tmp_normal, outside);
            if (t > 0.0 && t < tmax) return false;
        }
        else
        {
            for (int j = obj.triangleStart; j != obj.triangleEnd; j++)
            {
                const glm::ivec3& tri = dev_sceneInfo.modelInfo.dev_triangles[j];
                const glm::vec3& v0 = dev_sceneInfo.modelInfo.dev_vertices[tri[0]];
                const glm::vec3& v1 = dev_sceneInfo.modelInfo.dev_vertices[tri[1]];
                const glm::vec3& v2 = dev_sceneInfo.modelInfo.dev_vertices[tri[2]];
                t = triangleIntersectionTest(obj.Transform, v0, v1, v2, ray, tmp_intersect, tmp_normal, tmp_baryCoord);
                if (t > 0.0 && t < tmax) return false;
            }
        }

    }
    return true;
}



__device__ bool util_bvh_test_visibility(glm::vec3 p0, glm::vec3 p1, const SceneInfoDevice& sceneInfo)
{
    glm::vec3 dir = p1 - p0;
    if (glm::length(dir) < 0.001f) return true;
    Ray ray;
    ray.direction = glm::normalize(dir);
    ray.origin = p0;
    glm::vec3 t3 = (dir / ray.direction);
    float tmax = max(t3.x, max(t3.y, t3.z)) - 0.001f;
    int curr = bvh_get_near_child(sceneInfo.dev_bvhGPUArray, 0, ray.origin);

    ShadeableIntersection curr_intersect;
    curr_intersect.t = tmax;
    bool intersected = false;
    Bvh_traverse_state state = fromParent;

    while (curr >= 0 && curr < sceneInfo.bvhDataSize) {
        //printf("in loop");
        if (state == fromChild) {
            if (curr == 0) break;
            int parent = sceneInfo.dev_bvhGPUArray[curr].parent;
            if (curr == bvh_get_near_child(sceneInfo.dev_bvhGPUArray, parent, ray.origin)) {
                curr = bvh_get_sibling(sceneInfo.dev_bvhGPUArray, curr);
                state = fromSibling;
            }
            else {
                curr = parent;
                state = fromChild;
            }
        }
        else if (state == fromSibling) {
            if (!boundingBoxIntersectionTest(sceneInfo.dev_bvhGPUArray[curr].bbox, ray)) {
                curr = sceneInfo.dev_bvhGPUArray[curr].parent;
                state = fromChild;
            }
            else if (bvh_is_leaf(sceneInfo.dev_bvhGPUArray, curr)) {
                //printf("bvh has leaf in fromSibling");
                if (bvh_leaf_intersect(curr, ray, sceneInfo, &curr_intersect)) {
                    intersected = true;

                    //printf("now intersected \n");
                    //break;
                }
                curr = sceneInfo.dev_bvhGPUArray[curr].parent;
                state = fromChild;
            }
            else {
                curr = bvh_get_near_child(sceneInfo.dev_bvhGPUArray, curr, ray.origin);
                state = fromParent;
            }

        }
        else { //fromParent
            if (!boundingBoxIntersectionTest(sceneInfo.dev_bvhGPUArray[curr].bbox, ray)) {
                curr = bvh_get_sibling(sceneInfo.dev_bvhGPUArray, curr);
                state = fromSibling;
            }
            else if (bvh_is_leaf(sceneInfo.dev_bvhGPUArray, curr)) {
                //printf("bvh has leaf in fromParent");
                if (bvh_leaf_intersect(curr, ray, sceneInfo, &curr_intersect)) {
                    intersected = true;
                    //printf("now intersected \n");
                    //break;
                }
                curr = bvh_get_sibling(sceneInfo.dev_bvhGPUArray, curr);
                state = fromSibling;
            }
            else {
                curr = bvh_get_near_child(sceneInfo.dev_bvhGPUArray, curr, ray.origin);
                state = fromParent;
            }
        }
    }


    return intersected;
}
