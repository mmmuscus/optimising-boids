#include "framework.h"

void printMat4(mat4 mat) {
	printf("%f, %f, %f, %f\n", mat[0][0], mat[0][1], mat[0][2], mat[0][3]);
	printf("%f, %f, %f, %f\n", mat[1][0], mat[1][1], mat[1][2], mat[1][3]);
	printf("%f, %f, %f, %f\n", mat[2][0], mat[2][1], mat[2][2], mat[2][3]);
	printf("%f, %f, %f, %f\n\n", mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
}

void printVec3(vec3 vec) {
	printf("X: %f, Y: %f, Z: %f\n\n", vec.x, vec.y, vec.z);
}

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

struct VertexData {
	vec3 position, normal;

	//vec2 texcoord;
};

struct Light {
	vec3 La, Le, wLightPos;

	Light(vec3 _La, vec3 _Le, vec3 _wLightPos)
	{
		La = vec3(_La);
		Le = vec3(_Le);

		wLightPos = vec3(_wLightPos);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;

	Material(vec3 _kd, vec3 _ks, vec3 _ka, float _shininess)
	{
		kd = vec3(_kd);
		ks = vec3(_ks);
		ka = vec3(_ka);

		shininess = _shininess;
	}
};

struct RenderState {
	// Object related states
	mat4 M, Minv, V, P;

	// Camera related states
	vec3 wEye;

	// material related state thingies
	vec3 kd, ks, ka;
	float shininess;

	// light related state thingies
	vec3 La, Le, wLightPos;

	// texture related state thingies

public:

	void print()
	{
		printf("########################################\n");
		printf("---------------- OBJECT ----------------\n");

		printf("M:\n");
		printMat4(M);

		printf("Minv:\n");
		printMat4(Minv);

		printf("V:\n");
		printMat4(V);

		printf("P:\n");
		printMat4(P);

		printf("(MVP):\n");
		printMat4(M * V * P);

		printf("---------------- CAMERA ----------------\n");

		printf("wEye:\n");
		printVec3(wEye);

		printf("---------------- MATERIAL ----------------\n");

		printf("kd:\n");
		printVec3(kd);

		printf("ks:\n");
		printVec3(ks);

		printf("ka:\n");
		printVec3(ka);

		printf("shininess: %f\n", shininess);

		printf("---------------- LIGHT ----------------\n");

		printf("La\n");
		printVec3(La);

		printf("Le\n");
		printVec3(Le);

		printf("wLightPos\n");
		printVec3(wLightPos);
	}
};

struct Camera {
	vec3 wEye, wLookat, wVup;

	float fov = 75.0f * (float)M_PI / 180.0f;
	float asp = (float)windowWidth / windowHeight;
	float fp = 1;	// care for z fighting
	float bp = 30;

public:
	
	// view transformation matrix
	mat4 V()
	{
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);

		return TranslateMatrix(wEye * (-1)) * mat4(	u.x,	v.y,	w.x,	0,
													u.y,	v.y,	w.y,	0,
													u.z,	v.z,	w.z,	0,
													0,		0,		0,		1);
	}

	// projection matrix
	mat4 P()
	{
		return mat4(	1 / (tan(fov / 2) * asp),	0,					0,							0,
						0,							1 / tan(fov / 2),	0,							0,
						0,							0,					-(fp + bp) / (bp - fp),		-1,
						0,							0,					-2 * fp * bp / (bp - fp),	0);
	}
};

/*struct Texture {

};*/

class Geometry {
protected:

	unsigned int vao, vbo;

public:

	Geometry()
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
	}

	void Load(const std::vector<VertexData>& vtxData)
	{
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_DYNAMIC_DRAW);
		
		glEnableVertexAttribArray(0);	// position
		glEnableVertexAttribArray(1);	// normal
		//glEnableVertexAttribArray(2);	// texcoord

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		//glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	virtual void Draw() = 0;
	
	// hopefully this will be used sometime
	virtual void Animate(float t) {}

	~Geometry()
	{
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

struct ParamSurface : public Geometry {
	unsigned int VtxPerStrip, Strips;

public:

	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N = 30, int M = 30) 
	{
		VtxPerStrip = (M + 1) * 2;
		Strips = N;
		std::vector<VertexData> vtxData;

		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j <= M; j++)
			{
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}

		Load(vtxData);
	}

	void Draw()
	{
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < Strips; i++)
			glDrawArrays(GL_TRIANGLE_STRIP, i * VtxPerStrip, VtxPerStrip);
	}
};

struct Object {
	Geometry *geom;
	Material *mat;

	vec3 pos, rotationAxis;
	float rotationAngle;

public:

	Object(Geometry *_geom, Material *_mat, vec3 _pos)
	{
		geom = _geom;
		mat = _mat;

		pos = vec3(_pos);
		rotationAxis = vec3(0, 0, 1);
		rotationAngle = 0;
	}

	virtual void Draw(RenderState state)
	{
		state.M = RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(pos);
		state.Minv = TranslateMatrix(-pos) * RotationMatrix(-rotationAngle, rotationAxis);
		//state.texture = texture
		
		// MATERIAL TO STATE
		
		//shader->Bind(state);
		geom->Draw();
	}

	virtual void Animate(float tstart, float tend) { geom->Animate(tend); }
};

class Scene {

};

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
	float vertices[] = { -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f };
	glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
		sizeof(vertices),  // # bytes
		vertices,	      	// address
		GL_STATIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		0, NULL); 		     // stride, offset: tightly packed

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(gpuProgram.getId(), "color");
	glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
							  0, 0, 0, 1 };

	location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	glBindVertexArray(vao);  // Draw call
	glDrawArrays(GL_TRIANGLES, 0 /*startIdx*/, 3 /*# Elements*/);

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
/*void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}*/

// Key of ASCII code released
/*void onKeyboardUp(unsigned char key, int pX, int pY) {
}*/

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
