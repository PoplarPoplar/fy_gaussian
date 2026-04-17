import struct
import numpy as np

class BufferWriter:
    """
    Constructs a new instance of the class with either a specified size or an existing array.
    :param size: The number of points to be stored, each taking up 32 bytes. The buffer will be allocated 
                 with size * 32 bytes, plus extra space for the header.
    :param array: An existing numpy array or byte array whose buffer will be used.
    """
    def __init__(self, size=None, array=None):
        if size is not None:
            # Header size is 32 bytes; total buffer size = (size * 32) + 32 bytes
            self.buffer = bytearray(size * 32 + 32)  # Added 4 bytes for the "gssp" string
        elif array is not None:
            self.buffer = bytearray(array.tobytes())
        else:
            self.buffer = bytearray(32 + 32)  # Default to one point and header space
        self.view = memoryview(self.buffer)
        self.offset = 0

    def write_header(self, numbersplats, aabb_min, aabb_max):
        """
        Writes the header containing numbersplats and AABB min/max values to the buffer.
        :param numbersplats: The number of splats (as a uint32).
        :param aabb_min: A numpy array representing the min_point of the AABB.
        :param aabb_max: A numpy array representing the max_point of the AABB.
        """
        # Write "gssp" string (4 Uint8)
        struct.pack_into('<4B', self.buffer, self.offset, 103, 115, 115, 112)  # 'gssp' encoded in Uint8
        self.offset += 4

        struct.pack_into('<I', self.buffer, self.offset, numbersplats)  # numbersplats as Uint32
        self.offset += 4
        struct.pack_into('<3f', self.buffer, self.offset, *aabb_min)  # min_x, min_y, min_z as float
        self.offset += 12
        struct.pack_into('<3f', self.buffer, self.offset, *aabb_max)  # max_x, max_y, max_z as float
        self.offset += 12

    def write_point(self, p, cA, cB, C):
        """
        Writes a single point to the buffer.
        :param p: A numpy array with shape (3,) representing position (x, y, z).
        :param cA: A numpy array with shape (3,) representing cA (x, y, z).
        :param cB: A numpy array with shape (3,) representing cB (x, y, z).
        :param C: A numpy array with shape (4,) representing c (x, y, z, w).
        """
        # Ensure inputs are NumPy arrays
        p = np.asarray(p, dtype=np.float32)
        cA = np.asarray(cA, dtype=np.float32)
        cB = np.asarray(cB, dtype=np.float32)
        C = np.asarray(C, dtype=np.uint8)

        # Write position (3 Float32) and cB.x (1 Float32)
        struct.pack_into('<3f', self.buffer, self.offset, *p)
        self.offset += 12
        struct.pack_into('<f', self.buffer, self.offset, cB[0])
        self.offset += 4

        # Write cA.x (1 HalfFloat), cA.y (1 HalfFloat), cA.z (1 HalfFloat)
        # Write cB.y (1 HalfFloat), cB.z (1 HalfFloat)
        self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cA[0])
        self.offset += 2
        self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cA[1])
        self.offset += 2
        self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cA[2])
        self.offset += 2
        self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cB[1])
        self.offset += 2
        self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cB[2])
        self.offset += 2

        # Write empty half-float (16 bits)
        struct.pack_into('<H', self.buffer, self.offset, 0)
        self.offset += 2

        # Write c (4 Uint8)
        struct.pack_into('<4B', self.buffer, self.offset, *C)
        self.offset += 4
        
    def write_points(self, points, covAs, covBs, colors):
        """
        批量写入多个点到缓冲区
        :param points: (n, 3) numpy array, n 个点的坐标
        :param covAs: (n, 3) numpy array, n 个点的 cA 值
        :param covBs: (n, 3) numpy array, n 个点的 cB 值
        :param colors: (n, 4) numpy array, n 个点的颜色 (R, G, B, A)
        """
        n_points = points.shape[0]  # 获取要写入的点的数量
        
        for i in range(n_points):
            p = points[i]
            cA = covAs[i]
            cB = covBs[i]
            C = colors[i]

            # Write position (3 Float32) and cB.x (1 Float32)
            struct.pack_into('<3f', self.buffer, self.offset, *p)
            self.offset += 12
            struct.pack_into('<f', self.buffer, self.offset, cB[0])
            self.offset += 4

            # Write cA.x (1 HalfFloat), cA.y (1 HalfFloat), cA.z (1 HalfFloat)
            # Write cB.y (1 HalfFloat), cB.z (1 HalfFloat)
            self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cA[0])
            self.offset += 2
            self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cA[1])
            self.offset += 2
            self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cA[2])
            self.offset += 2
            self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cB[1])
            self.offset += 2
            self.buffer[self.offset:self.offset + 2] = struct.pack('<e', cB[2])
            self.offset += 2

            # Write empty half-float (16 bits)
            struct.pack_into('<H', self.buffer, self.offset, 0)
            self.offset += 2

            # Write c (4 Uint8)
            struct.pack_into('<4B', self.buffer, self.offset, *C)
            self.offset += 4

    def get_buffer(self):
        return self.buffer