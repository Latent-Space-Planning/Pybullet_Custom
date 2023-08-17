import numpy as np

def EPA(simplex, colliderA, colliderB):
    polytope = simplex.copy()
    faces = [
        0, 1, 2,
        0, 3, 1,
        0, 2, 3,
        1, 3, 2
    ]

    normals, minFace = get_face_normals(polytope, faces)

    minNormal = np.zeros(3)
    minDistance = np.finfo(float).max

    while minDistance == np.finfo(float).max:
        minNormal = normals[minFace][:3]
        minDistance = normals[minFace][3]

        support = support_function(colliderA, colliderB, minNormal)
        sDistance = np.dot(minNormal, support)

        if abs(sDistance - minDistance) > 0.001:
            minDistance = np.finfo(float).max

            uniqueEdges = []
            i = 0
            while i < len(normals):
                if same_direction(normals[i][:3], support):
                    f = i * 3

                    add_if_unique_edge(uniqueEdges, faces, f, f + 1)
                    add_if_unique_edge(uniqueEdges, faces, f + 1, f + 2)
                    add_if_unique_edge(uniqueEdges, faces, f + 2, f)

                    faces[f + 2] = faces[-1]
                    faces.pop()
                    faces[f + 1] = faces[-1]
                    faces.pop()
                    faces[f] = faces[-1]
                    faces.pop()

                    normals[i] = normals[-1]
                    normals = normals[:-1]

                    i -= 1

                i += 1

            newFaces = []
            for edgeIndex1, edgeIndex2 in uniqueEdges:
                newFaces.extend([edgeIndex1, edgeIndex2, len(polytope)])

            polytope.append(support)

            newNormals, newMinFace = get_face_normals(polytope, newFaces)

            oldMinDistance = np.finfo(float).max
            for i in range(len(normals)):
                if normals[i][3] < oldMinDistance:
                    oldMinDistance = normals[i][3]
                    minFace = i

            if newNormals[newMinFace][3] < oldMinDistance:
                minFace = newMinFace + len(normals)

            faces.extend(newFaces)
            normals = np.vstack((normals, newNormals))

    points = {}
    points["Normal"] = minNormal
    points["PenetrationDepth"] = minDistance + 0.001
    points["HasCollision"] = True

    return points


def get_face_normals(polytope, faces):
    normals = []
    minTriangle = 0
    minDistance = np.finfo(float).max

    for i in range(0, len(faces), 3):
        a = polytope[faces[i]]
        b = polytope[faces[i + 1]]
        c = polytope[faces[i + 2]]

        normal = np.cross(b - a, c - a)
        normal /= np.linalg.norm(normal)
        distance = np.dot(normal, a)

        if distance < 0:
            normal *= -1
            distance *= -1

        normals.append(np.array([normal[0], normal[1], normal[2], distance]))

        if distance < minDistance:
            minTriangle = i // 3
            minDistance = distance

    return normals, minTriangle


def add_if_unique_edge(edges, faces, a, b):
    reverse = next((edge for edge in edges if edge == (faces[b], faces[a])), None)

    if reverse:
        edges.remove(reverse)
    else:
        edges.append((faces[a], faces[b]))


def support_function(colliderA, colliderB, direction):
    # Implement your support function here
    pass


def same_direction(normal, support):
    return np.dot(normal, support) > 0


# Example usage
simplex = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0]), np.array([10.0, 11.0, 12.0])]
colliderA = None  # Replace with your collider A
colliderB = None  # Replace with your collider B

collision_points = EPA(simplex, colliderA, colliderB)
print("Collision Points:", collision_points)
