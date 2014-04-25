#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

#define PI 3.1415926535897932384626
#define IMPULSE_CAP 128

struct gnoise_params{ //struct for input params to gabor noise
	float a, density, filterSigma, octaves;
	vec4 sector;
	mat2 jacob;
};

struct gnoise_im_params{ //struct to pass intermediate values within gnoise
	mat2 filter, sigma_f_plus_g_inv;
	float ainv, a_prime_square, filt_scale;
};

//hash based on Blum, Blum & Shub 1986
//and Sharpe http://briansharpe.wordpress.com/2011/10/01/gpu-texture-free-noise/
const float bbsm = 1739.;//magic product of two primes chosen to have high period without float precision issues
vec2 bbsopt( const vec2 a ) {
	return fract( a * a * bbsm );
}
float bbsopt( const float a ) {
	return fract( a * a * bbsm );
}
vec2 mod1024(const vec2 v){
	return v - floor(v*0.0009765625)*1024.;
}
float seed(const vec2 p){
	vec2 h0 = bbsopt(p.xy*(1.0/bbsm)); //scale to small value for bbsopt; fp precision errors will make quasi infinite
	vec2 h1 = bbsopt(mod1024(p.xy)*(1.0/bbsm)+.5); //repeats every 1024 to destroy fp precision artifacts
	vec2 h2 = h0+h1;//best of both worlds
	return bbsopt(h2.y+bbsopt(h2.x));
}

//permutation polynomial
//based on Gustavson/McEwan https://github.com/ashima/webgl-noise/
//and Sharpe http://briansharpe.wordpress.com/2011/10/01/gpu-texture-free-noise/
// the permutation x <- 34x*x + x mod 289 maps 0 to 0.
// to prevent small values from getting stuck at zero, add a small constant term
//for efficiency, store all values in (0,1) and use x%y = fract(x/y)*y
const float pp_epsilon = (1./289.);
float nextRand(inout float u){//rng
	u = fract((u*34.0+1.0)*u+pp_epsilon);
	return fract((7.*289./288.)*u);
}

//approximate poisson distribution uniform u
//from Galerne, Lagae, Lefebvre, Drettakis
const float poisson_epsilon = .001;
int poisson(inout float u, const float m){
	float u1 = nextRand(u);
	float u2 = nextRand(u);
	float x = sqrt(-2.*log(u1+poisson_epsilon))*cos(2.*PI*u2);
	return int(m+sqrt(m)*x+.5);
}

//Gabor noise based on Lagae, Lefebvre, Drettakis, Dutre 2011
  float eval_cell(const vec2 cpos, const vec2 gpos, const vec2 dnbr, const gnoise_params params, const gnoise_im_params im_params){
	float u = seed(gpos+dnbr); //deterministic seed for nbr cell
	int impulses = poisson(u, params.density*(1./PI)); //number of impulses in nbr cell
	vec4 h = params.sector; //annular sector
	float a = params.a; //bandwidth
	float aps = im_params.a_prime_square; //intermediate calculations for filtering
	float filt_scale = im_params.filt_scale; //aps*im_params.ainv*im_params.ainv;
	vec2 fpos = cpos - dnbr;//fragment position in cell space
	
	float acc = 0.;
	//for impulses
	for(int k=0; k<IMPULSE_CAP; k++){
		if(k<impulses){
			//position of impulse in cell space - uniform distribution
			vec2 ipos = vec2(nextRand(u), nextRand(u));
			//displacement to fragment
			vec2 delta = (fpos - ipos)*im_params.ainv;
			//impulse frequency, orientation - distribution on input ranges
			float mfreq = pow(2., nextRand(u)*params.sector.y);
			float ifreq = h.x*mfreq; 
			float iorientation = mix(h.z-.5*h.w, h.z+.5*h.w, nextRand(u));
			//evaluate kernel, accumulate fragment value
			vec2 mu = ifreq*vec2(cos(iorientation), sin(iorientation));
			float phi = nextRand(u); //phase - uniform dist [0, 1]
			float filt_exp = -.5*dot(mu, im_params.sigma_f_plus_g_inv*mu);
			acc+= filt_scale/mfreq*exp(-PI*aps*dot(delta,delta)+filt_exp)*cos(2.*PI*(dot(delta, im_params.filter*mu)+phi));
		}else {break;}
	}
	return acc;
 }
 
 float det2x2(const mat2 m){
	return (m[0][0]*m[1][1] - m[0][1]*m[1][0]);
 }
 mat2 inv2x2(const mat2 m){
	return (1./det2x2(m))*mat2(m[1][1], -m[0][1], -m[1][0], m[0][0]);
 }
 mat2 id2x2(){
	return mat2(1.,0.,0.,1.);
 }
 
 //annular sector of pink noise
float gnoise(vec2 pos, const gnoise_params params) {
	gnoise_im_params im_params;
	im_params.ainv = 1./params.a;
	
	//compute positions for this fragment
	vec2 temp = pos*params.a; 
	vec2 cpos = fract(temp);
	vec2 gpos = floor(temp);
		
	mat2 jacob = params.jacob;
	mat2 jacob_t = mat2(jacob[0][0], jacob[1][0], jacob[0][1], jacob[1][1]);
	mat2 sigma_f_inv = (4.*PI*PI*params.filterSigma*params.filterSigma)*(jacob*jacob_t);
	mat2 sigma_f = inv2x2(sigma_f_inv);
	mat2 sigma_g_inv = (2.*PI*im_params.ainv*im_params.ainv)* id2x2();
	mat2 sigma_g = inv2x2(sigma_g_inv);
	mat2 sigma_fg_inv = sigma_f_inv + sigma_g_inv;
	mat2 sigma_fg = inv2x2(sigma_fg_inv);
	
	//filter params
	im_params.filter = sigma_fg * sigma_g_inv;	
	im_params.sigma_f_plus_g_inv = inv2x2(sigma_f + sigma_g);
	im_params.a_prime_square = 2.*PI*sqrt(det2x2(sigma_fg));
	im_params.filt_scale = im_params.a_prime_square*im_params.ainv*im_params.ainv;

	float value = 
		eval_cell(cpos, gpos, vec2(-1., -1.), params, im_params) +
		eval_cell(cpos, gpos, vec2(-1.,  0.), params, im_params) +
		eval_cell(cpos, gpos, vec2(-1.,  1.), params, im_params) +
		eval_cell(cpos, gpos, vec2( 0., -1.), params, im_params) +
		eval_cell(cpos, gpos, vec2( 0.,  0.), params, im_params) +
		eval_cell(cpos, gpos, vec2( 0.,  1.), params, im_params) +
		eval_cell(cpos, gpos, vec2( 1., -1.), params, im_params) +
		eval_cell(cpos, gpos, vec2( 1.,  0.), params, im_params) +
		eval_cell(cpos, gpos, vec2( 1.,  1.), params, im_params);
	
	
	//ad hoc attempt to normalize
	value*=.5*pow(params.density+1., -.5);
	float octexp = pow(2., params.sector.y);
	value*= (1.+params.sector.y)*octexp/(2.*octexp-1.);

	return value;

	
}
 
