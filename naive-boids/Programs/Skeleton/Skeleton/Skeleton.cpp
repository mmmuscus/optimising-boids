#include "framework.h"

void printMat4(mat4 mat) 
{
	printf("%f, %f, %f, %f\n", mat[0][0], mat[0][1], mat[0][2], mat[0][3]);
	printf("%f, %f, %f, %f\n", mat[1][0], mat[1][1], mat[1][2], mat[1][3]);
	printf("%f, %f, %f, %f\n", mat[2][0], mat[2][1], mat[2][2], mat[2][3]);
	printf("%f, %f, %f, %f\n", mat[3][0], mat[3][1], mat[3][2], mat[3][3]);
}

void printVec2(vec2 vec)
{
	printf("X: %f, Y: %f\n", vec.x, vec.y);
}

void printVec3(vec3 vec) 
{
	printf("X: %f, Y: %f, Z: %f\n", vec.x, vec.y, vec.z);
}

void printVec4(vec4 vec)
{
	printf("X: %f, Y: %f, Z: %f, W: %f\n", vec.x, vec.y, vec.z, vec.w);
}

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

// La is not properly calculated
// the shader is not good yet pls look at it thx
class PhongShader : public GPUProgram
{
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4 MVP, M, Minv; // MVP, Model, Model-inverse
		uniform vec3 wEye;         // pos of eye
		uniform vec3 wLightPos;		// light stuffz

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		//layout(location = 2) in vec2  vtxUV;				this is texture stuff

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight;		    // light dir in world space
		out float vtxDistanceFromLightSquaredVec;
		//out vec2 texcoord;

		out vec3 debug;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		    wView = wEye - (vec4(vtxPos, 1) * M).xyz;

			wLight = vtxPos - wLightPos;
			vtxDistanceFromLightSquaredVec = 1 / (pow(vtxPos.x - wLightPos.x, 2) + pow(vtxPos.y - wLightPos.y, 2) + pow(vtxPos.z - wLightPos.z, 2));
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    //texcoord = vtxUV;

			debug = vec3(1, 1, 1);
		}
	)";

	// fragment shader in GLSL
	// La is not properly calculated
	// THIS IS FOR DIRECTIONAL LIGHT NOT POSITIONAL !!!
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		//uniform sampler2D diffuseTexture;

		uniform vec3 kd, ks, ka;	// material stuffz
		uniform float shininess;	// shininess
		uniform vec3 La, Le;		// light stuffz

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight;        // the direction from where the light is coming from
		in	float vtxDistanceFromLightSquaredVec;
		//in  vec2 texcoord;

		in vec3 debug;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 L = normalize(wLight);
			vec3 H = normalize(L + V);
			float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
			fragmentColor = vec4(ka * La + (kd * cost + ks * pow(cosd, shininess)) * Le * vtxDistanceFromLightSquaredVec, 1);

			//fragmentColor = vec4(debug, 1);
		}
	)";

public:

	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		setUniform(state.M * state.V * state.P, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");

		setUniform(state.wEye, "wEye");

		setUniform(state.kd, "kd");
		setUniform(state.ks, "ks");
		setUniform(state.ka, "ka");
		setUniform(state.shininess, "shininess");

		setUniform(state.La, "La");
		setUniform(state.Le, "Le");
		setUniform(state.wLightPos, "wLightPos");

		// texture stuff
	}
};

PhongShader *shader;

struct VertexData {
	vec3 position, normal;

	//vec2 texcoord;
};

struct Light {
	vec3 La, Le, wLightPos;

	Light() {}

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

struct Camera {
	vec3 wEye, wLookat, wVup;

	float fov = 75.0f * (float)M_PI / 180.0f;
	float asp = (float)windowWidth / windowHeight;
	float fp = 1;	// care for z fighting
	float bp = 100;

public:

	void MoveWithKeys(bool a, bool w, bool s, bool d, float time)
	{
		vec3 dir = vec3(0, 0, 0);

		if (a) dir = dir + normalize(cross(wVup, wLookat - wEye));
		if (w) dir = dir + normalize(wLookat - wEye);
		if (s) dir = dir + normalize(wEye - wLookat);
		if (d) dir = dir + normalize(cross(wVup, wEye - wLookat));

		dir = dir * time;

		wLookat = wLookat + dir;
		wEye = wEye + dir;
	}

	void RotateWithMouse(vec2 start, vec2 end)
	{
		// just isolated for x
		float distanceX = end.x - start.x;
		float rotationAngleX = distanceX * fov / 2.0f;

		vec3 wLookatVector = normalize(wLookat - wEye);
		float wLookatDistance = length(wLookat - wEye);

		vec4 wLookatVector4 = vec4(wLookatVector.x, wLookatVector.y, wLookatVector.z, 1.0f) * RotationMatrix(rotationAngleX, wVup);
		wLookatVector = vec3(wLookatVector4.x, wLookatVector4.y, wLookatVector4.z) * wLookatDistance;

		wLookat = wEye + wLookatVector;

		// just isolated for y
		float distanceY = end.y - start.y;
		float rotationAngleY = distanceY * (fov / asp) / 2.0f;
		vec3 rotationAxisY = normalize(cross(wVup, wLookatVector));

		vec4 wVup4 = vec4(wVup.x, wVup.y, wVup.z, 1.0f) * RotationMatrix(rotationAngleY, rotationAxisY);
		wVup = vec3(wVup4.x, wVup4.y, wVup4.z);

		wLookatVector = normalize(wLookat - wEye);

		wLookatVector4 = vec4(wLookatVector.x, wLookatVector.y, wLookatVector.z, 1.0f) * RotationMatrix(rotationAngleY, rotationAxisY);
		wLookatVector = vec3(wLookatVector4.x, wLookatVector4.y, wLookatVector4.z) * wLookatDistance;

		wLookat = wEye + wLookatVector;
	}
	
	// view transformation matrix
	mat4 V()
	{
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);

		return TranslateMatrix(wEye * (-1)) * mat4(	u.x,	v.x,	w.x,	0,
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

// TEST PARAM SURFACE SPHERE
class Sphere : public ParamSurface
{
	float r;
public:
	Sphere(float _r)
	{
		r = _r;
		Create();
	}

	VertexData GenVertexData(float u, float v)
	{
		VertexData vd;
		float phi = u * 2 * M_PI;
		float theta = v * M_PI;

		vd.normal = vec3(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));
		vd.position = vd.normal * r;
		
		return vd;
	}
};

struct Object {
	Geometry *geom;
	Material *mat;

	vec3 pos, rotationAxis;
	float rotationAngle;

public:

	Object() {}

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
		state.kd = mat->kd;
		state.ks = mat->ks;
		state.ka = mat->ka;
		state.shininess = mat->shininess;

		shader->Bind(state);

		geom->Draw();
	}

	virtual void Animate(float tstart, float tend) { geom->Animate(tend); }
};

class Scene {
	Camera* camera;
	Light light;
	RenderState state;
	Object test;

	// objects we have;

public:

	Scene() {}

	Camera* getCamera() { return camera; }

	void Build()
	{
		shader = new PhongShader();

		camera = new Camera();

		camera->wEye = vec3(0, 0, 10);
		camera->wLookat = vec3(0, 0, 0);
		camera->wVup = vec3(0, 1, 0);

		light = Light(vec3(1, 1, 1), vec3(80, 80, 80), vec3(3, 3, -3));

		vec3 kd(0.3f, 0.2f, 0.1f);
		vec3 ks(0.008f, 0.008f, 0.008f);
		vec3 ka = kd * M_PI;
		float shininess = 10.0f;

		test = Object(new Sphere(3), new Material(kd, ks, ka, shininess), vec3(0, 0, 0));
	}

	void Render() 
	{
		// camera to state
		state.wEye = camera->wEye;
		state.V = camera->V();
		state.P = camera->P();

		// light to state
		state.La = light.La;
		state.Le = light.Le;
		state.wLightPos = light.wLightPos;

		// drawing objects
		test.Draw(state);
	}

	void Animate(float tstart, float tend)
	{

	}
};

Scene scene;

GPUProgram gpuProgram; // vertex and fragment shaders

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	scene.Build();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear frame buffer

	scene.Render();
	glutSwapBuffers();
}

bool aFlag = false;
bool wFlag = false;
bool sFlag = false;
bool dFlag = false;

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) 
{
	switch (key)
	{
		case 'a':
			aFlag = true;
			break;
		case 'w':
			wFlag = true;
			break;
		case 's':
			sFlag = true;
			break;
		case 'd':
			dFlag = true;
			break;
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY)
{
	switch (key)
	{
		case 'a':
			aFlag = false;
			break;
		case 'w':
			wFlag = false;
			break;
		case 's':
			sFlag = false;
			break;
		case 'd':
			dFlag = false;
			break;
	}
}

vec2 mouseStart = vec2(0, 0);
vec2 mouseEnd = vec2(0, 0);
bool moving = false;

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	mouseEnd = vec2(cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; mouseStart = vec2(cX, cY); mouseEnd = mouseStart; moving = true; break;
	case GLUT_UP:   buttonStat = "released"; moving = false; break;
	}
}

long lastTime = 0.0f;
long currTime = 0.0f;

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	currTime = glutGet(GLUT_ELAPSED_TIME) / 20.0f; // elapsed time since the start of the program
	float deltaTime = (float)currTime - (float)lastTime;
	lastTime = currTime;

	scene.getCamera()->MoveWithKeys(aFlag, wFlag, sFlag, dFlag, deltaTime);

	if (moving)
	{
		scene.getCamera()->RotateWithMouse(mouseStart, mouseEnd);
	}

	mouseStart = mouseEnd;

	glutPostRedisplay();
}
