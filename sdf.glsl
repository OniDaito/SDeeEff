#version 300 es

precision mediump float;

// Blackle's cool shader
// https://www.shadertoy.com/view/wtVyWK


uniform float   u_time;
uniform vec2    u_resolution;

in vec2 out_texcoord;
layout(location = 0) out vec4 out_colour; // out_color must be written in order to see anything

vec3 erot(vec3 p, vec3 ax, float ro) {
    return mix(dot(p,ax)*ax,p,cos(ro))+sin(ro)*cross(ax,p);
}

float scene(vec3 p) {
    //sdf is undefined outside the sphere, uncomment to witness the abominations
    if (length(p) > 10.) {
        return length(p)-0.8;
    }
    //neural networks can be really compact... when they want to be
vec4 f0_0=sin(p.y*vec4(-2.05,1.54,-3.36,-3.84)+p.z*vec4(.67,2.52,-1.76,3.27)+p.x*vec4(2.81,1.82,-.14,-1.22)+vec4(5.01,-.22,-6.58,1.69));
vec4 f0_1=sin(p.y*vec4(2.43,4.12,-1.73,3.03)+p.z*vec4(1.64,2.75,-1.98,-4.14)+p.x*vec4(-.12,3.69,2.73,-1.97)+vec4(7.71,7.05,-.45,2.38));
vec4 f0_2=sin(p.y*vec4(-.43,2.24,-1.54,-1.11)+p.z*vec4(-1.18,-.72,3.32,1.31)+p.x*vec4(-3.22,3.88,2.33,2.93)+vec4(-7.79,7.58,-3.17,-6.07));
vec4 f0_3=sin(p.y*vec4(-.69,-3.40,-3.06,2.53)+p.z*vec4(-2.40,-1.37,.21,2.75)+p.x*vec4(2.74,3.73,.66,.69)+vec4(6.49,4.53,.03,4.54));
vec4 f1_0=sin(mat4(.44,-.57,.09,.25,-.49,-.28,.09,-.07,-.01,-.46,.23,-.04,.30,.44,.70,-.07)*f0_0+
    mat4(-.37,-.53,-.47,-.71,.09,.02,.11,-.18,.20,-.31,.03,.14,.36,-.00,.47,-.36)*f0_1+
    mat4(.08,.08,-.12,-.10,.10,.24,.11,.17,.61,.01,.93,-.15,.46,-.05,.33,.59)*f0_2+
    mat4(.17,-.11,-.47,-.10,.12,-.09,-.11,.15,.51,-.21,.20,.77,.12,-.23,.61,-.54)*f0_3+
    vec4(2.21,.46,-1.69,2.09))/1.0+f0_0;
vec4 f1_1=sin(mat4(.06,.05,.17,-.32,-.31,.36,-.70,.51,-.11,-.38,-.18,-.53,-.69,.22,.59,.20)*f0_0+
    mat4(-.22,.56,.20,-.04,.20,.18,.00,.31,.36,1.10,-.36,.92,.23,1.12,.24,-.08)*f0_1+
    mat4(.30,.38,-.43,.31,.54,-.11,-.66,.57,-.76,.52,.35,-.15,.19,-.06,.46,.64)*f0_2+
    mat4(-1.26,-.32,-.04,-.14,.26,.29,-.98,.54,.74,-.35,1.20,.44,-.06,.17,.09,-.21)*f0_3+
    vec4(-3.34,.11,-1.74,.23))/1.0+f0_1;
vec4 f1_2=sin(mat4(.82,-.58,.09,.20,-.49,.47,.09,-.26,.51,-.13,.59,-.08,.79,-.41,.19,-.43)*f0_0+
    mat4(.48,.38,.05,-.58,-.66,.49,.01,-.45,-.13,-.14,.58,-.31,-.24,-.54,.32,.01)*f0_1+
    mat4(-.03,.18,-.05,.35,-.11,-.23,-.32,.05,-.07,-.15,.30,.17,-.46,.26,.25,-.43)*f0_2+
    mat4(.31,.77,.91,.21,-.11,.08,-.02,-.19,-.11,.11,.39,-.34,.87,.26,-.38,-.18)*f0_3+
    vec4(-.35,-.57,-2.63,.37))/1.0+f0_2;
vec4 f1_3=sin(mat4(.12,.14,-.22,-.18,.25,.01,.31,.36,-.13,.23,-.08,-.14,.04,.79,-.08,-.03)*f0_0+
    mat4(-.46,.22,.65,-.69,.43,-.10,.11,.26,-.72,-.75,.58,-.67,-.44,.07,.42,-.01)*f0_1+
    mat4(.41,-.42,-.07,-.08,-.46,-.16,.07,-.18,.08,.00,.29,.03,.52,.11,1.16,.41)*f0_2+
    mat4(-.07,.82,.08,.03,.05,.08,-.01,-.12,.54,-.05,.90,.43,.10,-.42,.01,-.46)*f0_3+
    vec4(-2.00,3.04,1.28,-1.85))/1.0+f0_3;
vec4 f2_0=sin(mat4(1.06,.04,.38,.18,.19,.38,-.10,.63,.55,-.13,1.74,-.49,.03,.04,.54,-.03)*f1_0+
    mat4(-.20,.50,.87,.36,-.40,-.36,-.17,.01,-.67,-.50,-.04,-.11,.43,-.24,.64,.25)*f1_1+
    mat4(-.11,.91,-.11,.38,.37,1.04,-.63,.20,-.42,-.77,.21,.20,-.22,.74,.93,-.05)*f1_2+
    mat4(-.10,.54,-.69,-.56,.28,1.65,.87,.49,.43,.37,.63,-.15,.18,-.57,.74,.50)*f1_3+
    vec4(-1.78,-1.02,-.57,.87))/1.4+f1_0;
vec4 f2_1=sin(mat4(-.13,.81,-.74,-1.63,-.19,.87,.48,-.06,-.58,.14,.14,-1.34,-.07,-.66,-1.42,.91)*f1_0+
    mat4(.53,-.33,-.11,-.69,-.45,-.37,.81,-.26,.23,-.54,-.02,.15,-.49,.38,.41,-.25)*f1_1+
    mat4(.09,.30,1.64,.54,-.16,-.27,-1.33,-.19,-.39,-.40,1.00,-.86,-.40,-.20,.03,-.52)*f1_2+
    mat4(-.60,.15,1.98,.97,.15,.85,-.39,.21,.42,-.16,.08,-.35,.59,-.01,-1.19,-.33)*f1_3+
    vec4(3.06,1.31,2.49,-1.67))/1.4+f1_1;
vec4 f2_2=sin(mat4(-.30,1.04,.89,-.45,.31,.74,-.03,.09,.88,-.25,.45,.54,.28,-.14,.33,-.63)*f1_0+
    mat4(1.26,-.64,.64,.48,.81,-.30,-.58,.44,.26,.08,-.46,-.21,.08,.24,.12,.07)*f1_1+
    mat4(-.82,.25,-.71,-.06,1.30,-.20,.32,-.29,.24,-.40,-.12,.80,1.11,-.02,-.39,-.18)*f1_2+
    mat4(.06,.02,-1.35,.21,.06,-.27,.33,-.02,.56,.23,.77,.00,2.63,.25,.73,.45)*f1_3+
    vec4(.71,-2.56,-2.34,-.69))/1.4+f1_2;
vec4 f2_3=sin(mat4(-.15,.69,-.61,-.41,.19,-.67,-.63,1.05,-.30,.94,.85,.30,-1.04,.41,.16,.02)*f1_0+
    mat4(.53,.46,.28,-.07,.57,-.09,.38,-.05,.17,-1.12,.08,.40,-.19,.14,-.54,-.60)*f1_1+
    mat4(-.00,-.89,-.62,.08,-.05,.12,-.47,-.20,-.12,.14,-.43,-.69,.29,.26,.39,-.61)*f1_2+
    mat4(-.50,1.21,-.03,-.40,-.35,.42,.70,-.01,.65,.45,.55,.20,-.03,-.07,.17,-.45)*f1_3+
    vec4(-.23,2.41,1.25,3.32))/1.4+f1_3;
return dot(f2_0,vec4(.11,.03,.03,.10))+
    dot(f2_1,vec4(-.05,.08,.03,-.04))+
    dot(f2_2,vec4(-.02,-.09,.06,.10))+
    dot(f2_3,vec4(.07,.06,.05,.07))+
    0.283;

}

vec3 norm(vec3 p) {
    mat3 k = mat3(p, p, p) - mat3(0.001);
    return normalize(scene(p) - vec3(scene(k[0]),scene(k[1]),scene(k[2])));
}

float sdSphere(vec3 p, float r) {
  return length(p) - r;
}

float ray_march(vec3 ro, vec3 rd, float start, float end) {
  float depth = start;

  for (int i = 0; i < 255; i++) {
    vec3 p = ro + depth * rd;
    float d = sdSphere(p, 1.);
    depth += d;
    if (d < 0.001 || depth > end) break;
  }

  return depth;
}


void main( )
{
    vec2 uv = (gl_FragCoord.xy-.5*u_resolution.xy)/u_resolution.y;
    vec3 ro = vec3(0, 0, 10.0); // ray origin that represents camera position
    vec3 rd = normalize(vec3(uv, -1)); // ray direction
    
    float yrot = -0.5;
    float zrot = u_time*.2;
   
    //ro = erot(ro, vec3(0,1,0), yrot);
    //init = erot(init, vec3(0,1,0), yrot);
    //cam = erot(cam, vec3(0,0,1), zrot);
    //init = erot(init, vec3(0,0,1), zrot);
    
    vec3 p = ro;
    bool hit = false;


    for (int i = 0; i < 150 && !hit; i++) {
        float dist = scene(erot(p,vec3(1,0,0), zrot));
        hit = dist*dist < 1e-6;
        p+=dist*rd;
        if (distance(p,ro)>20.) break;
    }   

    vec3 n = norm(p);
    vec3 r = reflect(ro,n);
    //don't ask how I stumbled on this texture
    vec3 nz = p; // - erot(p, vec3(1), 2.) + erot(p, vec3(1), 4.);
    float spec = length(sin(r*0.35+sin(nz*120.)*.15)*.4+.6)/sqrt(3.);
    spec *= smoothstep(-.3,.2,scene(p+r*.2));
    vec3 col = vec3(.1,.1,.12)*spec + pow(spec,8.);
    float bgdot = length(sin(ro*8.)*.4+.6)/2.;
    vec3 bg = vec3(.1,.1,.11) * bgdot + pow(bgdot, 10.);
    out_colour.xyz = hit ? col : bg;
    out_colour = smoothstep(-.02,1.05,sqrt(out_colour)) * (1.- dot(uv,uv)*.5);
    out_colour.w = 1.0;
}