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
    //sdf is undefined outside the unit sphere, uncomment to witness the abominations
    if (length(p) > 1.) {
        return length(p)-.8;
    }

    <--FRAGMENT-->
}

vec3 norm(vec3 p) {
    mat3 k = mat3(p, p, p) - mat3(0.001);
    return normalize(scene(p) - vec3(scene(k[0]),scene(k[1]),scene(k[2])));
}

void main( )
{
    vec2 uv = (gl_FragCoord.xy-.5*u_resolution.xy)/u_resolution.y;
    vec3 cam = normalize(vec3(1.5,uv));
    vec3 init = vec3(-3.,0,0);
    
    float yrot = 0.5;
    float zrot = u_time*.2;
   
    cam = erot(cam, vec3(0,1,0), yrot);
    init = erot(init, vec3(0,1,0), yrot);
    cam = erot(cam, vec3(0,0,1), zrot);
    init = erot(init, vec3(0,0,1), zrot);
    
    vec3 p = init;
    bool hit = false;
    for (int i = 0; i < 150 && !hit; i++) {
        float dist = scene(p);
        hit = dist*dist < 1e-6;
        p+=dist*cam;
        if (distance(p,init)>7.) break;
    }
    vec3 n = norm(p);
    vec3 r = reflect(cam,n);
    //don't ask how I stumbled on this texture
    vec3 nz = p - erot(p, vec3(1), 2.) + erot(p, vec3(1), 4.);
    float spec = length(sin(r*3.5+sin(nz*120.)*.15)*.4+.6)/sqrt(3.);
    spec *= smoothstep(-.3,.2,scene(p+r*.2));
    vec3 col = vec3(.1,.1,.12)*spec + pow(spec,8.);
    float bgdot = length(sin(cam*8.)*.4+.6)/2.;
    vec3 bg = vec3(.1,.1,.11) * bgdot + pow(bgdot, 10.);
    out_colour.xyz = hit ? col : bg;
    out_colour = smoothstep(-.02,1.05,sqrt(out_colour)) * (1.- dot(uv,uv)*.5);
    out_colour.w = 1.0;
}