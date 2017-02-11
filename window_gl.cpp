#ifndef CM_NOGL

#ifdef _WIN32
    #define WINDOWS_LEAN_AND_MEAN
    #define NOMINMAX
    #define _CRT_SECURE_NO_WARNINGS
    #include <windows.h>
    #define snprintf sprintf_s
#else
    #define VK_ESCAPE 27
#endif


#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "timing.h"
#include "window_gl.h"
#include "kernel.h"
#include "bmp_output.h"

static void DrawTexture();

// buffer handles/pointers
static GLuint pbo; // this PBO (pixel buffer object) is used to connect CUDA and OpenGL
static GLuint result_texture;  // the result is copied to this OpenGL texture
static uint32_t* image_buffer; // this is the actual memory pointer to the PBO

// Window & Image
#define CM_IMAGE_FORMAT GL_BGRA
static uint32_t window_width  = 1200;
static uint32_t window_height = 800;
static uint32_t sqrtSamples   = 1;
static GLint    maxTextureSize;

// Mandelbrot kernel parameters
static kernel_params params;

// main display loop
static void Display() {

    // clear the frame buffer
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // CUDA kernel timer
    cudaTimer t;
    t = startCudaTimer();

    // map OpenGL buffer (PBO) into CUDA address space
    cudaGLMapBufferObject((void**) &params.image_buffer, pbo);

    LaunchKernel(params);

    cudaGLUnmapBufferObject(pbo); // unmap buffer so OpenGL can use our data

    float gpu_time = stopCudaTimer(t);

    // copy the data from the PBO into our result texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo); 
    glBindTexture(GL_TEXTURE_2D, result_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, params.width, params.height, CM_IMAGE_FORMAT, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    
    DrawTexture();       // draw our result texture to the back buffer
    glutSwapBuffers();   // swap the window's frame buffers
    glutPostRedisplay(); // tell glut that our window has changed
    
    static char buf[256] = { 0 };
    snprintf(buf, sizeof(buf) - 1, "CUDA-Mandelbrot - GPU: %6.3fms - x: %e | y : %e | z : %e | i : %d", gpu_time, params.centerX, params.centerY, params.zoom, params.iter);
    glutSetWindowTitle(buf);
}

// display image to the screen as textured quad
static void DrawTexture() {

    // disable lighting and depth, enable textures
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    // set our view projection to orthographic, coordinates from -1,-1,-1 to 1, 1, 1
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    
    // render a screen sized rectangle
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_QUADS);              // begin a quadrilateral
    glTexCoord2f(0.0, 0.0);         // bottom left
    glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0);         // bottom right
    glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0);         // top right
    glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0);         // top left
    glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
}

static void WriteImageToDisk(const char* filename, bool writeLargeImage = false) {
    uint32_t bufferSize;
    uint32_t *cpuBuffer;
    int result;

    if (writeLargeImage) {
        // writes the entire image buffer to disk, possibly hundreds of MB!
        bufferSize = params.width * params.height * sizeof(*image_buffer);
        cpuBuffer = (uint32_t*) malloc(bufferSize);
        cudaGLMapBufferObject((void**) &image_buffer, pbo);
        cudaMemcpy(cpuBuffer, image_buffer, bufferSize, cudaMemcpyDeviceToHost);
        cudaGLUnmapBufferObject(pbo);
        result = write_bmp(filename, params.width, params.height, (uint8_t*) cpuBuffer);
    }
    else {
        // writes the window contents to disk
        bufferSize = window_width * window_height * sizeof(*image_buffer);
        cpuBuffer = (uint32_t*) malloc(bufferSize);
        glReadPixels(0, 0, window_width, window_height, GL_BGRA, GL_UNSIGNED_BYTE, cpuBuffer);
        result = write_bmp(filename, window_width, window_height, (uint8_t*) cpuBuffer);
    }

    free(cpuBuffer);
    if (result == 0) {
        printf("Wrote image to %s\n", filename);
    }
}


// initialize the PBO and texture for transferring data from CUDA to OpenGL
static void InitGLBuffers() {

    size_t num_texels = params.width * params.height;
    size_t size_tex_data = sizeof(*image_buffer) * num_texels;

    // create buffer object
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_ARRAY_BUFFER, pbo);
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, NULL, GL_DYNAMIC_DRAW); // set buffer size

    // register this buffer object with CUDA
    cudaGLRegisterBufferObject(pbo);

    // create the texture that we use to display the result
    glGenTextures(1, &result_texture);
    glBindTexture(GL_TEXTURE_2D, result_texture);

    // set texture filtering (linear required to simulate supersampling)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // set texture size and format
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, params.width, params.height, 0, CM_IMAGE_FORMAT, GL_UNSIGNED_BYTE, NULL);
}

// delete and recreate buffer
static void ResizeImageBuffer() {
    params.width  = window_width  * sqrtSamples;
    params.height = window_height * sqrtSamples;
    cudaGLUnregisterBufferObject(pbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &result_texture);
    InitGLBuffers();
}

static void WindowResizeCallback(int width, int height) {
    // set new OpenGL view port
    glViewport(0, 0, width, height);
    window_width = width;
    window_height = height;

    ResizeImageBuffer();
}

///////////
// INPUT //
///////////

static double zoomSpeed = 1.1;
static double startModSpeed = 0.025;
static boolean lmbState = GLUT_UP;
static boolean rmbState = GLUT_UP;
static int lastX; // last mouse X position
static int lastY; // last mouse Y position

// only called while a mouse button is down
static void MouseDragCallback(int x, int y) {
    
    // set scale so that the mouse position stays constant in the mandelbrot coordinate system
    double scale = (2.0 / double(window_height)) * params.zoom;

    if (lmbState == GLUT_DOWN) {
        params.centerX -= (x - lastX) * scale;
        params.centerY += (y - lastY) * scale;
    }
    if (rmbState == GLUT_DOWN) {
        params.startX += (x - lastX) * scale;
        params.startY -= (y - lastY) * scale;
    }
    lastX = x;
    lastY = y;
}

static void MouseClickCallback(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        lmbState = state;
        lastX = x;
        lastY = y;
    }
    if (button == GLUT_RIGHT_BUTTON) {
        rmbState = state;
        lastX = x;
        lastY = y;
    }
}

static void MouseWheelCallback(int wheel, int direction, int x, int y) {
    if (direction == -1)
        params.zoom *= zoomSpeed;
    else
        params.zoom *= 1.0 / zoomSpeed;
}

static void KeyboardCallback(unsigned char key, int x, int y) {

    switch (key) {
    case VK_ESCAPE:
        glutLeaveMainLoop(); // exit
        break;
    case 'e':
    case '-':
        params.zoom *= zoomSpeed;
        break;
    case 'q':
    case '+':
        params.zoom *= 1.0 / zoomSpeed;
        break;
    case 'w':
        params.startY += startModSpeed * params.zoom;
        break;
    case 's':
        params.startY -= startModSpeed * params.zoom;
        break;
    case 'a':
        params.startX -= startModSpeed * params.zoom;
        break;
    case 'd':
        params.startX += startModSpeed * params.zoom;
        break;
    case 'c':
        params.maxlen2 *= 1.0 / zoomSpeed;
        break;
    case 'v':
        params.maxlen2 *= zoomSpeed;
        break;
    case ',':
        params.exponent -= 0.01 * params.zoom;
        break;
    case '.':
        params.exponent += 0.01 * params.zoom;
        break;
    case 'k':
        zoomSpeed -= 0.01;
        break;
    case 'l':
        zoomSpeed += 0.01;
        break;
    case 'f':
        params.iter *= 1.0 / zoomSpeed;
        if (params.iter == 0) params.iter = 1;
        break;
    case 'g':
        params.iter = ceil(params.iter * zoomSpeed);
        break;
    case 'n': {
        uint32_t newSpp = sqrtSamples << 1;
        if (((window_width  * newSpp) < (uint32_t) maxTextureSize) &&
            ((window_height * newSpp) < (uint32_t) maxTextureSize)) {

            sqrtSamples = newSpp;
            ResizeImageBuffer();
        }
    } break;
    case 'b':
        if (sqrtSamples > 1) {
            sqrtSamples >>= 1;
            ResizeImageBuffer();
        }
        break;
    case 'r': { // reset params
        params = kernel_params();
        sqrtSamples = 1;
        ResizeImageBuffer();
    } break;
    case 'p': {
        static char filename[64];
        static uint32_t prefix = uint32_t(getTime() ^ (getTime() >> 32));
        static int fileCounter = 0;
        snprintf(filename, sizeof(filename) - 1, "output_%d_%d.bmp", prefix, fileCounter++);
        WriteImageToDisk(filename);
    } break;
    }
    zoomSpeed = fmax(zoomSpeed, 1.01);
}

static void APIENTRY PrintDebugMessage(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam) {
    printf("%s\n", message);
}

bool InitGLWindow(int argc, const char *argv[]) {

    // create OpenGL context and window
#if _DEBUG
    glutInitContextFlags(GLUT_DEBUG);
#endif
    glutInit(&argc, (char**) argv);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA-Mandelbrot");
    
    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object GL_EXT_framebuffer_object ")) {
        printf("ERROR: Support for necessary OpenGL extensions missing.");
        return false;
    }

    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTextureSize);

    params.width  = window_width  * sqrtSamples;
    params.height = window_height * sqrtSamples;

    InitGLBuffers();

    // register callbacks
    glutDisplayFunc(Display);
    glutReshapeFunc(WindowResizeCallback);
    glutKeyboardFunc(KeyboardCallback);
    glutMouseWheelFunc(MouseWheelCallback);
    glutMouseFunc(MouseClickCallback);
    glutMotionFunc(MouseDragCallback);
    
#if _DEBUG
    printf("  VENDOR: %s\n", (const char *) glGetString(GL_VENDOR));
    printf(" VERSION: %s\n", (const char *) glGetString(GL_VERSION));
    printf("RENDERER: %s\n", (const char *) glGetString(GL_RENDERER));

    // register debug callback (either OpenGL 4.3 or GL_ARB_debug_output is required)
    if (glewIsSupported("GL_VERSION_4_3")) {
        glDebugMessageCallback((GLDEBUGPROC) printDebugMessage, NULL);
        // set debug output to be called in the same thread so we can set a breakpoint
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    }
    else if (glewIsSupported("GL_ARB_debug_output")) {
        glDebugMessageCallbackARB((GLDEBUGPROC) printDebugMessage, NULL);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
    }
#endif

    return true;
}

#endif