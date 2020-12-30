Shader "UmutBebek/URP/ShaderToy/girl"
{
   Properties
   {
      _Channel0("Channel0 (RGB)", 2D) = "" {}
      _Channel1("Channel1 (RGB)", 2D) = "" {}
      _Channel2("Channel2 (RGB)", 2D) = "" {}
      [HideInInspector]iMouse("Mouse", Vector) = (0,0,0,0)
      /*_Iteration("Iteration", float) = 1
      _NeighbourPixels("Neighbour Pixels", float) = 1
      _Lod("Lod",float) = 0
      _AR("AR Mode",float) = 0*/
      AA("AA", float) = 1

   }

   SubShader
   {
      // With SRP we introduce a new "RenderPipeline" tag in Subshader. This allows to create shaders
      // that can match multiple render pipelines. If a RenderPipeline tag is not set it will match
      // any render pipeline. In case you want your subshader to only run in LWRP set the tag to
      // "UniversalRenderPipeline"
      Tags{"RenderType" = "Opaque" "RenderPipeline" = "UniversalRenderPipeline" "IgnoreProjector" = "True"}
      LOD 300

      // ------------------------------------------------------------------
      // Forward pass. Shades GI, emission, fog and all lights in a single pass.
      // Compared to Builtin pipeline forward renderer, LWRP forward renderer will
      // render a scene with multiple lights with less drawcalls and less overdraw.
      Pass
      {
         // "Lightmode" tag must be "UniversalForward" or not be defined in order for
         // to render objects.
         Name "StandardLit"
         //Tags{"LightMode" = "UniversalForward"}

         //Blend[_SrcBlend][_DstBlend]
         //ZWrite Off ZTest Always
         //ZWrite[_ZWrite]
         //Cull[_Cull]

         HLSLPROGRAM
         // Required to compile gles 2.0 with standard SRP library
         // All shaders must be compiled with HLSLcc and currently only gles is not using HLSLcc by default
         #pragma prefer_hlslcc gles
         #pragma exclude_renderers d3d11_9x
         #pragma exclude_renderers d3d9
         #pragma target 3.0         

         //--------------------------------------
         // GPU Instancing
         #pragma multi_compile_instancing

         #pragma vertex LitPassVertex
         #pragma fragment LitPassFragment

         #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
         #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/CommonMaterial.hlsl"
         #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
         //do not add LitInput, it has already BaseMap etc. definitions, we do not need them (manually described below)
         //#include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"

         float4 _Channel0_ST;
         TEXTURE2D(_Channel0);       SAMPLER(sampler_Channel0);
         float4 _Channel1_ST;
         TEXTURE2D(_Channel1);       SAMPLER(sampler_Channel1);
         float4 _Channel2_ST;
         TEXTURE2D(_Channel2);       SAMPLER(sampler_Channel2);

         float4 iMouse;
         float AA;

         /*float _Lod;
         float _Iteration;
         float _NeighbourPixels;
         float _AR;*/

         struct Attributes
         {
            float4 positionOS   : POSITION;
            float2 uv           : TEXCOORD0;
            UNITY_VERTEX_INPUT_INSTANCE_ID
         };

         struct Varyings
         {
            float2 uv                       : TEXCOORD0;
            float4 positionCS               : SV_POSITION;
            float4 screenPos                : TEXCOORD1;
         };

         Varyings LitPassVertex(Attributes input)
         {
            Varyings output;

            // VertexPositionInputs contains position in multiple spaces (world, view, homogeneous clip space)
            // Our compiler will strip all unused references (say you don't use view space).
            // Therefore there is more flexibility at no additional cost with this struct.
            VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);

            // TRANSFORM_TEX is the same as the old shader library.
            output.uv = TRANSFORM_TEX(input.uv, _Channel0);
            // We just use the homogeneous clip position from the vertex input
            output.positionCS = vertexInput.positionCS;
            output.screenPos = ComputeScreenPos(vertexInput.positionCS);
            return output;
         }

         #define FLT_MAX 3.402823466e+38
         #define FLT_MIN 1.175494351e-38
         #define DBL_MAX 1.7976931348623158e+308
         #define DBL_MIN 2.2250738585072014e-308

         #define iTimeDelta unity_DeltaTime.x
         // float;

         #define iFrame ((int)(_Time.y / iTimeDelta))
         // int;

         #define clamp(x,minVal,maxVal) min(max(x, minVal), maxVal)

         float mod(float a, float b)
         {
            return a - floor(a / b) * b;
         }
         float2 mod(float2 a, float2 b)
         {
            return a - floor(a / b) * b;
         }
         float3 mod(float3 a, float3 b)
         {
            return a - floor(a / b) * b;
         }
         float4 mod(float4 a, float4 b)
         {
            return a - floor(a / b) * b;
         }

         // Created by inigo quilez - iq / 2019 
         // License Creative Commons Attribution - NonCommercial - ShareAlike 3.0 Unported License. 
         // 
         // 
         // An animation test - a happy and blobby creature jumping and 
         // looking around. It gets off - model very often , but it looks 
         // good enough I think. 
         // 
         // Making - of and related math / shader / art explanations ( 6 hours 
         // long ) : https: // www.youtube.com / watch?v = Cfe5UQ - 1L9Q 
         // 
         // Video capture: https: // www.youtube.com / watch?v = s_UOFo2IULQ 


         #if HW_PERFORMANCE == 0
            #define AA 1
         #else 
            #define AA 2 // Set AA to 1 if your machine is too slow 
         #endif

         // Basic utility functions (sdfs, noises, shaping functions)
         // and also the camera setup which is shaded between the
         // background rendering code ("Buffer A" tab) and the character
         // rendering code ("Image" tab)

         // http://iquilezles.org/www/articles/smin/smin.htm
         float smin( float a, float b, float k )
         {
            float h = max(k-abs(a-b),0.0);
            return min(a, b) - h*h*0.25/k;
         }

         // http://iquilezles.org/www/articles/smin/smin.htm
         float smax( float a, float b, float k )
         {
            k *= 1.4;
            float h = max(k-abs(a-b),0.0);
            return max(a, b) + h*h*h/(6.0*k*k);
         }

         // http://iquilezles.org/www/articles/smin/smin.htm
         float smin3( float a, float b, float k )
         {
            k *= 1.4;
            float h = max(k-abs(a-b),0.0);
            return min(a, b) - h*h*h/(6.0*k*k);
         }

         float sclamp(in float x, in float a, in float b )
         {
            float k = 0.1;
            return smax(smin(x,b,k),a,k);
         }

         // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
         float opOnion( in float sdf, in float thickness )
         {
            return abs(sdf)-thickness;
         }

         // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
         float opRepLim( in float p, in float s, in float lima, in float limb )
         {
            return p-s*clamp(round(p/s),lima,limb);
         }


         float det( float2 a, float2 b ) { return a.x*b.y-b.x*a.y; }
         float ndot(float2 a, float2 b ) { return a.x*b.x-a.y*b.y; }
         float dot2( in float2 v ) { return dot(v,v); }
         float dot2( in float3 v ) { return dot(v,v); }


         // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
         float sdTorus( in float3 p, in float ra, in float rb )
         {
            return length( float2(length(p.xz)-ra,p.y) )-rb;
         }

         // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
         float sdCappedTorus(in float3 p, in float2 sc, in float ra, in float rb)
         {
            p.x = abs(p.x);
            float k = (sc.y*p.x>sc.x*p.z) ? dot(p.xz,sc) : length(p.xz);
            return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
         }

         // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
         float sdSphere( in float3 p, in float r ) 
         {
            return length(p)-r;
         }

         // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
         float sdEllipsoid( in float3 p, in float3 r ) 
         {
            float k0 = length(p/r);
            float k1 = length(p/(r*r));
            return k0*(k0-1.0)/k1;
         }

         // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
         float sdBox( in float3 p, in float3 b )
         {
            float3 d = abs(p) - b;
            return min( max(max(d.x,d.y),d.z),0.0) + length(max(d,0.0));
         }

         // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
         float sdArc( in float2 p, in float2 scb, in float ra )
         {
            p.x = abs(p.x);
            float k = (scb.y*p.x>scb.x*p.y) ? dot(p.xy,scb) : length(p.xy);
            return sqrt( dot(p,p) + ra*ra - 2.0*ra*k );
         }

         #if 1
            // http://research.microsoft.com/en-us/um/people/hoppe/ravg.pdf
            // { dist, t, y (above the plane of the curve, x (away from curve in the plane of the curve))
               float4 sdBezier( float3 p, float3 va, float3 vb, float3 vc )
               {
                  float3 w = normalize( cross( vc-vb, va-vb ) );
                  float3 u = normalize( vc-vb );
                  float3 v =          ( cross( w, u ) );
                  //----  
                  float2 m = float2( dot(va-vb,u), dot(va-vb,v) );
                  float2 n = float2( dot(vc-vb,u), dot(vc-vb,v) );
                  float3 q = float3( dot( p-vb,u), dot( p-vb,v), dot(p-vb,w) );
                  //----  
                  float mn = det(m,n);
                  float mq = det(m,q.xy);
                  float nq = det(n,q.xy);
                  //----  
                  float2  g = (nq+mq+mn)*n + (nq+mq-mn)*m;
                  float f = (nq-mq+mn)*(nq-mq+mn) + 4.0*mq*nq;
                  float2  z = 0.5*f*float2(-g.y,g.x)/dot(g,g);
                  //float t = clamp(0.5+0.5*(det(z,m+n)+mq+nq)/mn, 0.0 ,1.0 );
                  float t = clamp(0.5+0.5*(det(z-q.xy,m+n))/mn, 0.0 ,1.0 );
                  float2 cp = m*(1.0-t)*(1.0-t) + n*t*t - q.xy;
                  //----  
                  float d2 = dot(cp,cp);
                  return float4(sqrt(d2+q.z*q.z), t, q.z, -sign(f)*sqrt(d2) );
               }
            #else
               float det( float3 a, float3 b, in float3 v ) { return dot(v,cross(a,b)); }

               // my adaptation to 3d of http://research.microsoft.com/en-us/um/people/hoppe/ravg.pdf
               // { dist, t, y (above the plane of the curve, x (away from curve in the plane of the curve))
                  float4 sdBezier( float3 p, float3 b0, float3 b1, float3 b2 )
                  {
                     b0 -= p;
                     b1 -= p;
                     b2 -= p;
                     
                     float3  d21 = b2-b1;
                     float3  d10 = b1-b0;
                     float3  d20 = (b2-b0)*0.5;

                     float3  n = normalize(cross(d10,d21));

                     float a = det(b0,b2,n);
                     float b = det(b1,b0,n);
                     float d = det(b2,b1,n);
                     float3  g = b*d21 + d*d10 + a*d20;
                     float f = a*a*0.25-b*d;

                     float3  z = cross(b0,n) + f*g/dot(g,g);
                     float t = clamp( dot(z,d10-d20)/(a+b+d), 0.0 ,1.0 );
                     float3 q = lerp(lerp(b0,b1,t), lerp(b1,b2,t),t);
                     
                     float k = dot(q,n);
                     return float4(length(q),t,-k,-sign(f)*length(q-n*k));
                  }
               #endif

               // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
               float2 sdSegment(float3 p, float3 a, float3 b)
               {
                  float3 pa = p-a, ba = b-a;
                  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
                  return float2( length( pa - ba*h ), h );
               }

               // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
               float2 sdSegmentOri(float2 p, float2 b)
               {
                  float h = clamp( dot(p,b)/dot(b,b), 0.0, 1.0 );
                  return float2( length( p - b*h ), h );
               }

               // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
               float sdFakeRoundCone(float3 p, float b, float r1, float r2)
               {
                  float h = clamp( p.y/b, 0.0, 1.0 );
                  p.y -= b*h;
                  return length(p) - lerp(r1,r2,h);
               }

               // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
               float sdCone( in float3 p, in float2 c )
               {
                  float2 q = float2( length(p.xz), p.y );

                  float2 a = q - c*clamp( (q.x*c.x+q.y*c.y)/dot(c,c), 0.0, 1.0 );
                  float2 b = q - c*float2( clamp( q.x/c.x, 0.0, 1.0 ), 1.0 );
                  
                  float s = -sign( c.y );
                  float2 d = min( float2( dot( a, a ), s*(q.x*c.y-q.y*c.x) ),
                  float2( dot( b, b ), s*(q.y-c.y)  ));
                  return -sqrt(d.x)*sign(d.y);
               }

               // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
               float sdRhombus(float3 p, float la, float lb, float h, float ra)
               {
                  p = abs(p);
                  float2 b = float2(la,lb);
                  float f = clamp( (ndot(b,b-2.0*p.xz))/dot(b,b), -1.0, 1.0 );
                  float2 q = float2(length(p.xz-0.5*b*float2(1.0-f,1.0+f))*sign(p.x*b.y+p.z*b.x-b.x*b.y)-ra, p.y-h);
                  return min(max(q.x,q.y),0.0) + length(max(q,0.0));
               }

               // http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
               float4 opElongate( in float3 p, in float3 h )
               {
                  float3 q = abs(p)-h;
                  return float4( max(q,0.0), min(max(q.x,max(q.y,q.z)),0.0) );
               }
               //-----------------------------------------------

               // ray-infinite-cylinder intersection
               float2 iCylinderY( in float3 ro, in float3 rd, in float rad )
               {
                  float3 oc = ro;
                  float a = dot( rd.xz, rd.xz );
                  float b = dot( oc.xz, rd.xz );
                  float c = dot( oc.xz, oc.xz ) - rad*rad;
                  float h = b*b - a*c;
                  if( h<0.0 ) return float2(-1.0,-1.0);
                  h = sqrt(h);
                  return float2(-b-h,-b+h)/a;
               }

               // ray-infinite-cone intersection
               float2 iConeY(in float3 ro, in float3 rd, in float k )
               {
                  float a = dot(rd.xz,rd.xz) - k*rd.y*rd.y;
                  float b = dot(ro.xz,rd.xz) - k*ro.y*rd.y;
                  float c = dot(ro.xz,ro.xz) - k*ro.y*ro.y; 
                  
                  float h = b*b-a*c;
                  if( h<0.0 ) return float2(-1.0,-1.0);
                  h = sqrt(h);
                  return float2(-b-h,-b+h)/a;
               }

               //-----------------------------------------------

               float linearstep(float a, float b, in float x )
               {
                  return clamp( (x-a)/(b-a), 0.0, 1.0 );
               }

               float2 rot( in float2 p, in float an )
               {
                  float cc = cos(an);
                  float ss = sin(an);
                  return mul(float2x2(cc,-ss,ss,cc),p);
               }

               float expSustainedImpulse( float t, float f, float k )
               {
                  return smoothstep(0.0,f,t)*1.1 - 0.1*exp2(-k*max(t-f,0.0));
               }

               float bnoise( in float x )
               {
                  float i = floor(x);
                  float f = frac(x);
                  float s = sign(frac(x/2.0)-0.5);
                  float k = 0.5+0.5*sin(i);
                  return s*f*(f-1.0)*((16.0*k-4.0)*f*(f-1.0)-1.0);
               }

               float3 fbm13( in float x, in float g )
               {    
                  float3 n = float3(0,0,0);
                  float s = 1.0;
                  for( int i=0; i<6; i++ )
                  {
                     n += s*float3(bnoise(x),bnoise(x+13.314),bnoise(x+31.7211));
                     s *= g;
                     x *= 2.01;
                     x += 0.131;
                  }
                  return n;
               }


               //--------------------------------------------------
               //const float X1 = 1.6180339887498948; const float H1 = float( 1.0/X1 );
               //const float X2 = 1.3247179572447460; const float2  H2 = float2(  1.0/X2, 1.0/(X2*X2) );
               //const float X3 = 1.2207440846057595; const float3  H3 = float3(  1.0/X3, 1.0/(X3*X3), 1.0/(X3*X3*X3) );


               //--------------------------------------
               float3x3 calcCamera( in float time, out float3 oRo, out float oFl )
               {
                  float3 ta = float3( 0.0, -0.3, 0.0 );
                  float3 ro = float3( -0.5563, -0.2, 2.7442 );
                  float fl = 1.7;

                  float3 fb1 = fbm13( 0.15*time, 0.50 );
                  //ro.xyz += 0.010*fb1.xyz;
                  float3 fb2 = fbm13( 0.33*time, 0.65 );
                  fb2 = fb2*fb2*sign(fb2);
                  //ta.xy += 0.005*fb2.xy;
                  float cr = -0.01 + 0.002*fb2.z;
                  cr = 0;                 
                  
                  // camera matrix
                  float3 ww = normalize( ta - ro );
                  float3 uu = normalize( cross(ww,float3(sin(cr),cos(cr),0.0) ) );
                  float3 vv =          ( cross(uu,ww));
                  
                  oRo = ro;
                  oFl = fl;

                  return float3x3(uu,vv,ww);
               }

               #define ZERO min(iFrame,0)
               #define ZEROU min(uint(iFrame),0u)

               // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

               #define ZEROExtended ( min ( iFrame , 0 ) )            


               // This SDF is really 6 braids at once (through domain
               // repetition) with three strands each (brute forced)
               float4 sdHair( float3 p, float3 pa, float3 pb, float3 pc, float an, out float2 occ_id) 
               {
                  float4 b = sdBezier(p, pa,pb,pc );
                  float2 q = rot(b.zw,an);
                  
                  float2 id2 = round(q/0.1);
                  id2 = clamp(id2,float2(0,0),float2(2,1));
                  q -= 0.1*id2;

                  float id = 11.0*id2.x + id2.y*13.0;

                  q += smoothstep(0.5,0.8,b.y)*0.02*float2(0.4,1.5)*cos( 23.0*b.y + id*float2(13,17));

                  occ_id.x = clamp(length(q)*8.0-0.2,0.0,1.0);
                  float4 res = float4(99,q,b.y);
                  for( int i=0; i<3; i++ )
                  {
                     float2 tmp = q + 0.01*cos( id + 180.0*b.y + float2(2*i,6-2*i));
                     float lt = length(tmp)-0.02;
                     if( lt<res.x )
                     { 
                        occ_id.y = id+float(i); 
                        res.x = lt; 
                        res.yz = tmp;
                     }
                  }
                  return res;
               }

               // The SDF for the hoodie and jacket. It's a very distorted
               // ellipsoid, torus section, a segment and a sphere.
               float4 sdHoodie( in float3 pos )
               {
                  float3 opos = pos;

                  pos.x   += 0.09*sin(3.5*pos.y-0.5)*sin(    pos.z) + 0.015;
                  pos.xyz += 0.03*sin(2.0*pos.y)*sin(7.0*pos.zyx);
                  
                  // hoodie
                  float3 hos = pos-float3(0.0,-0.33,0.15);
                  hos.x -= 0.031*smoothstep(0.0,1.0,opos.y+0.33);
                  hos.yz = rot(hos.yz,0.9);
                  float d1 = sdEllipsoid(hos,float3(0.96-pos.y*0.1,1.23,1.5));
                  float d2 = 0.95*pos.z-0.312*pos.y-0.9;
                  float d = max(opOnion(d1,0.01), d2 );
                  
                  // shoulders
                  float3 sos = float3( abs(pos.x), pos.yz );    
                  float2 se = sdSegment(sos, float3(0.18,-1.6,-0.3), float3(1.1,-1.9,0.0) );
                  d = smin(d,se.x-lerp(0.25,0.43,se.y),0.4);
                  d = smin(d,sdSphere(sos-float3(0.3,-2.2,0.4), 0.5 ),0.2);

                  // neck
                  opos.x -= 0.02*sin(9.0*opos.y);
                  float4 w = opElongate( opos-float3(0.0,-1.2,0.3), float3(0.0,0.3,0.0) );
                  d = smin(d,
                  w.w+sdCappedTorus(float3(w.xy,-w.z),float2(0.6,-0.8),0.6,0.02),
                  0.1);
                  
                  // bumps
                  d += 0.004*sin(pos.x*90.0)*sin(pos.y*90.0)*sin(pos.z*90.0);
                  d -= 0.002*sin(pos.x*300.0);
                  d -= 0.02*(1.0-smoothstep(0.0,0.04,abs(opOnion(pos.x,1.1))));
                  
                  // border
                  d = min(d,length(float2(d1,d2))-0.015);
                  
                  return float4(d,pos);
               }

               // moves the head (and hair and hoodie). This could be done
               // more efficiently (with a single matrix or quaternion),
               // but this code was optimized for editing, not for runtime
               float3 moveHead( in float3 pos, in float3 an, in float amount)
               {
                  pos.y -= -1.0;
                  pos.xz = rot(pos.xz,amount*an.x);
                  pos.xy = rot(pos.xy,amount*an.y);
                  pos.yz = rot(pos.yz,amount*an.z);
                  pos.y += -1.0;
                  return pos;
               }

               // the animation state
               float3 animData; // { blink, nose follow up, mouth } 
               float3 animHead; // { head rotation angles }

               // SDF of the girl. It is not as efficient as it should, 
               // both in terms of performance and euclideanness of the
               // returned distance. Among other things I tweaked the
               // overal shape of the head though scaling right in the
               // middle of the design process (see 1.02 and 1.04 numbers
               // below). I should have backpropagated those adjustements
               // to the  primitives themselves, but I didn't and now it's
               // too late. So, I am paying some cost there.
               //
               // Generally, she is modeled to camera (her face's shape 
               // looks bad from other perspectives. She's made of five
               // ellipsoids blended together for the face, a cone and
               // three spheres for the nose, a torus for the teeh and two
               // quadratic curves for the lips. The neck is a cylinder,
               // the hair is made of three quadratic that are repeated
               // multiple times through domain repetition and each of
               // them contains three more curves in order to make the
               // braids. The hoodie is an ellipsoid deformed with
               // two sine waves and cut in half, the neck is an elongated
               // torus section and the shoulders are capsules.
               //
               float4 map( in float3 pos, in float time, out float outMat, out float3 uvw )
               {
                  outMat = 1.0;

                  float3 oriPos = pos;
                  
                  // head deformation and transformation
                  pos.y /= 1.04;
                  float3 opos;
                  opos = moveHead( pos, animHead, smoothstep(-1.2, 0.2,pos.y) );
                  pos  = moveHead( pos, animHead, smoothstep(-1.4,-1.0,pos.y) );
                  pos.x *= 1.04;
                  pos.y /= 1.02;
                  uvw = pos;

                  // symmetric coord systems (sharp, and smooth)
                  float3 qos = float3(abs(pos.x),pos.yz);
                  float3 sos = float3(sqrt(qos.x*qos.x+0.0005),pos.yz);                  
                  
                  // head
                  float d = sdEllipsoid( pos-float3(0.0,0.05,0.07), float3(0.8,0.75,0.85) );

                  // jaw
                  float3 mos = pos-float3(0.0,-0.38,0.35); mos.yz = rot(mos.yz,0.4);
                  mos.yz = rot(mos.yz,0.1*animData.z);
                  float d2 = sdEllipsoid(mos-float3(0,-0.17,0.16),
                  float3(0.66+sclamp(mos.y*0.9-0.1*mos.z,-0.3,0.4),
                  0.43+sclamp(mos.y*0.5,-0.5,0.2),
                  0.50+sclamp(mos.y*0.3,-0.45,0.5)));
                  
                  // mouth hole
                  d2 = smax(d2,-sdEllipsoid(mos-float3(0,0.06,0.6+0.05*animData.z), float3(0.16,0.035+0.05*animData.z,0.1)),0.01);
                  
                  // lower lip    
                  float4 b = sdBezier(float3(abs(mos.x),mos.yz), 
                  float3(0,0.01,0.61),
                  float3(0.094+0.01*animData.z,0.015,0.61),
                  float3(0.18-0.02*animData.z,0.06+animData.z*0.05,0.57-0.006*animData.z));
                  float isLip = smoothstep(0.045,0.04,b.x+b.y*0.03);
                  d2 = smin(d2,b.x - 0.027*(1.0-b.y*b.y)*smoothstep(1.0,0.4,b.y),0.02);
                  d = smin(d,d2,0.19);

                  // chicks
                  d = smin(d,sdSphere(qos-float3(0.2,-0.33,0.62),0.28 ),0.04);
                  
                  // who needs ears
                  

                  // eye sockets
                  float3 eos = sos-float3(0.3,-0.04,0.7);
                  eos.xz = rot(eos.xz,-0.2);
                  eos.xy = rot(eos.xy,0.3);
                  eos.yz = rot(eos.yz,-0.2);
                  d2 = sdEllipsoid( eos-float3(-0.05,-0.05,0.2), float3(0.20,0.14-0.06*animData.x,0.1) );
                  d = smax( d, -d2, 0.15 );

                  eos = sos-float3(0.32,-0.08,0.8);
                  eos.xz = rot(eos.xz,-0.4);
                  d2 = sdEllipsoid( eos, float3(0.154,0.11,0.1) );
                  d = smax( d, -d2, 0.05 );

                  float3 oos = qos - float3(0.25,-0.06,0.42);
                  
                  // eyelid
                  d2 = sdSphere( oos, 0.4 );
                  oos.xz = rot(oos.xz, -0.2);
                  oos.xy = rot(oos.xy, 0.2);
                  float3 tos = oos;        
                  oos.yz = rot(oos.yz,-0.6+0.58*animData.x);

                  //eyebags
                  tos = tos-float3(-0.02,0.06,0.2+0.02*animData.x);
                  tos.yz = rot(tos.yz,0.8);
                  tos.xy = rot(tos.xy,-0.2);
                  d = smin( d, sdTorus(tos,0.29,0.01), 0.03 );
                  
                  // eyelids
                  eos = qos - float3(0.33,-0.07,0.53);
                  eos.xy = rot(eos.xy, 0.2);
                  eos.yz = rot(eos.yz,0.35-0.25*animData.x);
                  d2 = smax(d2-0.005, -max(oos.y+0.098,-eos.y-0.025), 0.02 );
                  d = smin( d, d2, 0.012 );

                  // eyelashes
                  oos.x -= 0.01;
                  float xx = clamp( oos.x+0.17,0.0,1.0);
                  float ra = 0.35 + 0.1*sqrt(xx/0.2)*(1.0-smoothstep(0.3,0.4,xx))*(0.6+0.4*sin(xx*256.0));
                  float rc = 0.18/(1.0-0.7*smoothstep(0.15,0.5,animData.x));
                  oos.y -= -0.18 - (rc-0.18)/1.8;
                  d2 = (1.0/1.8)*sdArc( oos.xy*float2(1.0,1.8), float2(0.9,sqrt(1.0-0.9*0.9)), rc )-0.001;
                  float deyelashes = max(d2,length(oos.xz)-ra)-0.003;
                  
                  // nose
                  eos = pos-float3(0.0,-0.079+animData.y*0.005,0.86);
                  eos.yz = rot(eos.yz,-0.23);
                  float h = smoothstep(0.0,0.26,-eos.y);
                  d2 = sdCone( eos-float3(0.0,-0.02,0.0), float2(0.03,-0.25) )-0.04*h-0.01;
                  eos.x = sqrt(eos.x*eos.x + 0.001);
                  d2 = smin( d2, sdSphere(eos-float3(0.0, -0.25,0.037),0.06 ), 0.07 );
                  d2 = smin( d2, sdSphere(eos-float3(0.1, -0.27,0.03 ),0.04 ), 0.07 );
                  d2 = smin( d2, sdSphere(eos-float3(0.0, -0.32,0.05 ),0.025), 0.04 );        
                  d2 = smax( d2,-sdSphere(eos-float3(0.07,-0.31,0.038),0.02 ), 0.035 );
                  d = smin(d,d2,0.05-0.03*h);
                  
                  // mouth
                  eos = pos-float3(0.0,-0.38+animData.y*0.003+0.01*animData.z,0.71);
                  tos = eos-float3(0.0,-0.13,0.06);
                  tos.yz = rot(tos.yz,0.2);
                  float dTeeth = sdTorus(tos,0.15,0.015);
                  eos.yz = rot(eos.yz,-0.5);
                  eos.x /= 1.04;

                  // nose-to-upperlip connection
                  d2 = sdCone( eos-float3(0,0,0.03), float2(0.14,-0.2) )-0.03;
                  d2 = max(d2,-(eos.z-0.03));
                  d = smin(d,d2,0.05);

                  // upper lip
                  eos.x = abs(eos.x);
                  b = sdBezier(eos, float3(0.00,-0.22,0.17),
                  float3(0.08,-0.22,0.17),
                  float3(0.17-0.02*animData.z,-0.24-0.01*animData.z,0.08));
                  d2 = length(b.zw/float2(0.5,1.0)) - 0.03*clamp(1.0-b.y*b.y,0.0,1.0);
                  d = smin(d,d2,0.02);
                  isLip = max(isLip,(smoothstep(0.03,0.005,abs(b.z+0.015+abs(eos.x)*0.04))
                  -smoothstep(0.45,0.47,eos.x-eos.y*1.15)));

                  // valley under nose
                  float2 se = sdSegment(pos, float3(0.0,-0.45,1.01),  float3(0.0,-0.47,1.09) );
                  d2 = se.x-0.03-0.06*se.y;
                  d = smax(d,-d2,0.04);
                  isLip *= smoothstep(0.01,0.03,d2);

                  // neck
                  se = sdSegment(pos, float3(0.0,-0.65,0.0), float3(0.0,-1.7,-0.1) );
                  d2 = se.x - 0.38;

                  // shoulders
                  se = sdSegment(sos, float3(0.0,-1.55,0.0), float3(0.6,-1.65,0.0) );
                  d2 = smin(d2,se.x-0.21,0.1);
                  d = smin(d,d2,0.4);
                  
                  // register eyelases now
                  float4 res = float4( d, isLip, 0, 0 );
                  //float4 res = float4( d, 0, 0, 0 );

                  if( deyelashes<res.x )
                  {
                     res.x = deyelashes*0.8;
                     res.yzw = float3(0.0,1.0,0.0);
                  }
                  // register teeth now
                  if( dTeeth<res.x )
                  {
                     res.x = dTeeth;
                     outMat = 5.0;
                  }
                  
                  // eyes
                  pos.x /=1.05;        
                  eos = qos-float3(0.25,-0.06,0.42);
                  d2 = sdSphere(eos,0.4);
                  if( d2<res.x ) 
                  { 
                     res.x = d2;
                     outMat = 2.0;
                     uvw = pos;
                  }
                  
                  // hair
                  {
                     float2 occ_id, tmp;
                     qos = pos; qos.x=abs(pos.x);

                     float4 pres = sdHair(pos,float3(-0.3, 0.55,0.8), 
                     float3( 0.95, 0.7,0.85), 
                     float3( 0.4,-1.45,0.95),
                     -0.9,occ_id);

                     float4 pres2 = sdHair(pos,float3(-0.4, 0.6,0.55), 
                     float3(-1.0, 0.4,0.2), 
                     float3(-0.6,-1.4,0.7),
                     0.6,tmp);
                     if( pres2.x<pres.x ) { pres=pres2; occ_id=tmp;  occ_id.y+=40.0;}

                     pres2 = sdHair(qos,float3( 0.4, 0.7,0.4), 
                     float3( 1.0, 0.5,0.45), 
                     float3( 0.4,-1.45,0.55),
                     -0.2,tmp);
                     if( pres2.x<pres.x ) { pres=pres2; occ_id=tmp;  occ_id.y+=80.0;}
                     

                     pres.x *= 0.8;
                     if( pres.x<res.x )
                     {
                        res = float4( pres.x, occ_id.y, 0.0, occ_id.x );
                        uvw = pres.yzw;
                        outMat = 4.0;
                     }
                  }

                  // hoodie
                  float4 tmp = sdHoodie( opos );
                  if( tmp.x<res.x )
                  {
                     res.x = tmp.x;
                     outMat = 3.0;
                     uvw  = tmp.yzw;
                  }

                  return res;
               }

               // SDF of the girl again, but with extra high frequency
               // modeling detail. While the previous one is used for
               // raymarching and shadowing, this one is used for normal
               // computation. This separation is conceptually equivalent
               // to decoupling detail from base geometry with "normal
               // maps", but done in 3D and with SDFs, which is way
               // simpler and can be done corretly (something rarely seen
               // in 3D engines) without any complexity.
               float4 mapD( in float3 pos, in float time )
               {
                  float matID;
                  float3 uvw;
                  float4 h = map(pos, time, matID, uvw);
                  
                  if( matID<1.5 ) // skin
                  {
                     // pores
                     float d = 0;
                     h.x += 0.0015*d*d;
                  }
                  else if( matID>3.5 && matID<4.5 ) // hair
                  {
                     // some random displacement to evoke hairs
                     float te = 0;
                     h.x -= 0.02*te;
                  }    
                  return h;
               }

               // Computes the normal of the girl's surface (the gradient
               // of the SDF). The implementation is weird because of the
               // technicalities of the WebGL API that forces us to do
               // some trick to prevent code unrolling. More info here:
               //
               // http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
               //
               float3 calcNormal( in float3 pos, in float time )
               {
                  const float eps = 0.001;
                  #if 0    
                     float2 e = float2(1.0,-1.0)*0.5773;
                     return normalize( e.xyy*map( pos + e.xyy*eps,time,kk ).x + 
                     e.yyx*map( pos + e.yyx*eps,time,kk ).x + 
                     e.yxy*map( pos + e.yxy*eps,time,kk ).x + 
                     e.xxx*map( pos + e.xxx*eps,time,kk ).x );
                  #else
                     float4 n = float4(0,0,0,0);
                     for( int i=0; i<4; i++ )
                     {
                        float4 s = float4(pos, 0.0);
                        float kk; float3 kk2;
                        s[i] += eps;
                        n[i] = mapD(s.xyz, time).x;
                        //if( n.x+n.y+n.z+n.w>100.0 ) break;
                     }
                     return normalize(n.xyz-n.w);
                  #endif   
               }

               // Compute soft shadows for a given light, with a single
               // ray insead of using montecarlo integration or shadowmap
               // blurring. More info here:
               //
               // http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
               //
               float calcSoftshadow( in float3 ro, in float3 rd, in float mint, in float tmax, in float time, float k )
               {
                  // first things first - let's do a bounding volume test
                  float2 sph = iCylinderY( ro, rd, 1.5 );
                  //float2 sph = iConeY(ro-float3(-0.05,3.7,0.35),rd,0.08);
                  tmax = min(tmax,sph.y);

                  // raymarch and track penumbra    
                  float res = 1.0;
                  float t = mint;
                  for( int i=0; i<128; i++ )
                  {
                     float kk; float3 kk2;
                     float h = map( ro + rd*t, time, kk, kk2 ).x;
                     res = min( res, k*h/t );
                     t += clamp( h, 0.005, 0.1 );
                     if( res<0.002 || t>tmax ) break;
                  }
                  return max( res, 0.0 );
               }

               // Computes convexity for our girl SDF, which can be used
               // to approximate ambient occlusion. More info here:
               //
               // https://iquilezles.org/www/material/nvscene2008/rwwtt.pdf
               //
               float calcOcclusion( in float3 pos, in float3 nor, in float time )
               {
                  const float X4 = 1.1673039782614187; 
                  const float4 H4 = float4(  1.0/X4, 1.0/(X4*X4), 1.0/(X4*X4*X4), 1.0/(X4*X4*X4*X4) );
                  float kk; float3 kk2;
                  float ao = 0.0;
                  //float off = textureLod(iChannel3,gl_FragCoord.xy/256.0,0.0).x;
                  float off = 0;
                  float4 k = float4(0.7012912,0.3941462,0.8294585,0.109841)+off;
                  for( int i=ZERO; i<16; i++ )
                  {
                     k = frac(k + H4);
                     float3 ap = normalize(-1.0+2.0*k.xyz);
                     float h = k.w*0.1;
                     ap = (nor+ap)*h;
                     float d = map( pos+ap, time, kk, kk2 ).x;
                     ao += max(0.0,h-d);
                     if( ao>16.0 ) break;
                  }
                  ao /= 16.0;
                  return clamp( 1.0-ao*24.0, 0.0, 1.0 );
               }

               // Computes the intersection point between our girl SDF and
               // a ray (coming form the camera in this case). It's a
               // traditional and basic/uncomplicated SDF raymarcher. More
               // info here:
               //
               // https://iquilezles.org/www/material/nvscene2008/rwwtt.pdf
               //
               float2 intersect( in float3 ro, in float3 rd, in float tmax, in float time, out float3 cma, out float3 uvw )
               {
                  cma = float3(0,0,0);
                  uvw = float3(0,0,0);
                  float matID = -1.0;

                  float t = 1.0;

                  // bounding volume test first
                  float2 sph = iCylinderY( ro, rd, 1.5 );
                  //float2 sph = iConeY(ro-float3(-0.05,3.7,0.35),rd,0.08);
                  
                  if( sph.y<0.0 ) 
                  {
                     return float2(-1.0,-1.0);
                  }                
                  
                  // clip raymarch space to bonding volume
                  tmax = min(tmax,sph.y);
                  t    = max(1.0, sph.x);
                  tmax = 20;
                  t = 1;
                  
                  // raymarch
                  for( int i=0; i<256; i++ )
                  {
                     float3 pos = ro + t*rd;

                     float tmp;
                     float4 h = map(pos,time,tmp,uvw);
                     if( h.x<0.001 )
                     {
                        cma = h.yzw;
                        matID = tmp;
                        break;
                     }
                     t += h.x*0.95;
                     if( t>tmax ) break;
                  }

                  return float2(t,matID);
               }

               // This is a replacement for a traditional dot(N,L) diffuse
               // lobe (called N.L in teh code) that fake some subsurface
               // scattering (transmision of light thorugh the skin that
               // surfaces as a red glow)
               //
               float3 sdif( float ndl, float ir )
               {
                  float pndl = clamp( ndl, 0.0, 1.0 );
                  float nndl = clamp(-ndl, 0.0, 1.0 );
                  return float3(pndl,pndl,pndl) + float3(1.0,0.1,0.01)*0.7*pow(clamp(ir*0.75-nndl,0.0,1.0),2.0);
               }

               // Animates the eye central position (not the actual random
               // darts). It's carefuly synched with the head motion, to
               // make the eyes anticipate the head turn (without this
               // anticipation, the eyes and the head are disconnected and
               // it all looks like a zombie/animatronic)
               //
               float animEye( in float time )
               {
                  const float w = 6.1;
                  float t = mod(time-0.31,w*1.0);
                  
                  float q = frac((time-0.31)/(2.0*w));
                  float s = (q > 0.5) ? 1.0 : 0.0;
                  return (t<0.15)?1.0-s:s;
               }

               // Renders the girl. It finds the ray-girl intersection
               // point, computes the normal at the intersection point,
               // computes the ambient occlusion approximation, does per
               // material setup (color, specularity, subsurface
               // coefficient and paints some fake occlusion), and finally
               // does the lighting computations.
               //
               // Lighting is not based on pathtracing. Instead the bounce
               // lighting occlusion signals are created manually (placed
               // and sized by hand). The subsurface scattering in the
               // nose area is also painted by hand. There's not much
               // attention to the physicall correctness of the light
               // response and materials, but generally all signal do
               // follow physically based rendering practices.
               //
               float3 renderGirl( in float2 p, in float3 ro, in float3 rd, in float tmax, in float3 col, in float time )
               {
                  // --------------------------
                  // find ray-girl intersection
                  // --------------------------
                  float3 cma, uvw;
                  float2 tm = intersect( ro, rd, tmax, time, cma, uvw );

                  // --------------------------
                  // shading/lighting	
                  // --------------------------
                  if( tm.y>0.0 )
                  {                     
                     float3 pos = ro + tm.x*rd;
                     float3 nor = calcNormal(pos, time);

                     float ks = 1.0;
                     float se = 16.0;
                     float tinterShadow = 0.0;
                     float sss = 0.0;
                     float focc = 1.0;
                     //float frsha = 1.0;

                     // --------------------------
                     // material
                     // --------------------------
                     if( tm.y<1.5 ) // skin
                     {
                        float3 qos = float3(abs(uvw.x),uvw.yz);

                        // base skin color
                        col = lerp(float3(0.225,0.15,0.12),
                        float3(0.24,0.1,0.066),
                        smoothstep(0.4 ,0.0,length( qos.xy-float2(0.42,-0.3)))+
                        smoothstep(0.15,0.0,length((qos.xy-float2(0,-0.29))/float2(1.4,1))));
                        
                        // fix that ugly highlight
                        col -= 0.03*smoothstep(0.13,0.0,length((qos.xy-float2(0,-0.49))/float2(2,1)));
                        
                        // lips
                        col = lerp(col,float3(0.14,0.06,0.1),cma.x*step(-0.7,qos.y));
                        
                        // eyelashes
                        col = lerp(col,float3(0.04,0.02,0.02)*0.6,0.9*cma.y);

                        // fake skin drag
                        uvw.y += 0.025*animData.x*smoothstep(0.3,0.1,length(uvw-float3(0.0,0.1,1.0)));
                        uvw.y -= 0.005*animData.y*smoothstep(0.09,0.0,abs(length((uvw.xy-float2(0.0,-0.38))/float2(2.5,1.0))-0.12));
                        
                        // freckles
                        float2 ti = floor(9.0+uvw.xy/0.04);
                        float2 uv = frac(uvw.xy/0.04)-0.5;
                        float te = frac(111.0*sin(1111.0*ti.x+1331.0*ti.y));
                        te = smoothstep(0.9,1.0,te)*exp(-dot(uv,uv)*24.0); 
                        col *= lerp(float3(1.1,1.1,1.1),float3(0.8,0.6,0.4), te);

                        // texture for specular
                        //ks = 0.5 + 4.0*texture(iChannel3,uvw.xy*1.1).x;
                        ks = 0;
                        se = 12.0;
                        ks *= 0.5;
                        tinterShadow = 1.0;
                        sss = 1.0;
                        ks *= 1.0 + cma.x;
                        
                        // black top
                        col *= 1.0-smoothstep(0.48,0.51,uvw.y);
                        
                        // makeup
                        float d2 = sdEllipsoid(qos-float3(0.25,-0.03,0.43),float3(0.37,0.42,0.4));
                        col = lerp(col,float3(0.06,0.024,0.06),1.0 - smoothstep(0.0,0.03,d2));

                        // eyebrows
                        {
                           #if 0
                              // youtube video version
                              float4 be = sdBezier( qos, float3(0.165+0.01*animData.x,0.105-0.02*animData.x,0.89),
                              float3(0.37,0.18-0.005*animData.x,0.82+0.005*animData.x), 
                              float3(0.53,0.15,0.69) );
                              float ra = 0.005 + 0.015*sqrt(be.y);
                           #else
                              // fixed version
                              float4 be = sdBezier( qos, float3(0.16+0.01*animData.x,0.11-0.02*animData.x,0.89),
                              float3(0.37,0.18-0.005*animData.x,0.82+0.005*animData.x), 
                              float3(0.53,0.15,0.69) );
                              float ra = 0.005 + 0.01*sqrt(1.0-be.y);
                           #endif
                           float dd = 1.0+0.05*(0.7*sin((sin(qos.x*3.0)/3.0-0.5*qos.y)*350.0)+
                           0.3*sin((qos.x-0.8*qos.y)*250.0+1.0));
                           float d = be.x - ra*dd;
                           float mask = 1.0-smoothstep(-0.005,0.01,d);
                           col = lerp(col,float3(0.04,0.02,0.02),mask*dd );
                        }

                        // fake occlusion
                        focc = 0.2+0.8*pow(1.0-smoothstep(-0.4,1.0,uvw.y),2.0);
                        focc *= 0.5+0.5*smoothstep(-1.5,-0.75,uvw.y);
                        focc *= 1.0-smoothstep(0.4,0.75,abs(uvw.x));
                        focc *= 1.0-0.4*smoothstep(0.2,0.5,uvw.y);
                        
                        focc *= 1.0-smoothstep(1.0,1.3,1.7*uvw.y-uvw.x);
                        
                        //frsha = 0.0;
                     }
                     else if( tm.y<2.5 ) // eye
                     {
                        // The eyes are fake in that they aren't 3D. Instead I simply
                        // stamp a 2D mathematical drawing of an iris and pupil. That
                        // includes the highlight and occlusion in the eyesballs.
                        
                        sss = 1.0;

                        float3 qos = float3(abs(uvw.x),uvw.yz);
                        float ss = sign(uvw.x);
                        
                        // iris animation
                        float dt = floor(time*1.1);
                        float ft = frac(time*1.1);
                        float2 da0 = sin(1.7*(dt+0.0)) + sin(2.3*(dt+0.0)+float2(1.0,2.0));
                        float2 da1 = sin(1.7*(dt+1.0)) + sin(2.3*(dt+1.0)+float2(1.0,2.0));
                        float2 da = lerp(da0,da1,smoothstep(0.9,1.0,ft));

                        float gg = animEye(time);
                        da *= 1.0+0.5*gg;
                        qos.yz = rot(qos.yz,da.y*0.004-0.01);
                        qos.xz = rot(qos.xz,da.x*0.004*ss-gg*ss*(0.03-step(0.0,ss)*0.014)+0.02);

                        float3 eos = qos-float3(0.31,-0.055 - 0.03*animData.x,0.45);
                        
                        // iris
                        float r = length(eos.xy)+0.005;
                        float a = atan2(eos.y,ss*eos.x);
                        float3 iris = float3(0.09,0.0315,0.0135);
                        iris += iris*3.0*(1.0-smoothstep(0.0,1.0, abs((a+3.14159)-2.5) ));
                        //iris *= 0.35+0.7*texture(iChannel2,float2(r,a/6.2831)).x;
                        iris *= 0.35;
                        // base color
                        col = float3(0.42,0.42,0.42);
                        col *= 0.1+0.9*smoothstep(0.10,0.114,r);
                        col = lerp( col, iris, 1.0-smoothstep(0.095,0.10,r) );
                        col *= smoothstep(0.05,0.07,r);
                        
                        // fake occlusion backed in
                        float edis = length((float2(abs(uvw.x),uvw.y)-float2(0.31,-0.07))/float2(1.3,1.0));
                        col *= lerp( float3(1,1,1), float3(0.4,0.2,0.1), linearstep(0.07,0.16,edis) );

                        // fake highlight
                        qos = float3(abs(uvw.x),uvw.yz);
                        col += (0.5-gg*0.3)*(1.0-smoothstep(0.0,0.02,length(qos.xy-float2(0.29-0.05*ss,0.0))));
                        
                        se = 128.0;

                        // fake occlusion
                        focc = 0.2+0.8*pow(1.0-smoothstep(-0.4,1.0,uvw.y),2.0);
                        focc *= 1.0-linearstep(0.10,0.17,edis);
                        //frsha = 0.0;
                     }
                     else if( tm.y<3.5 )// hoodie
                     {
                        sss = 0.0;
                        //col = float3(0.81*texture(iChannel0,uvw*6.0).x);
                        col = 0;
                        ks *= 2.0;
                        
                        // logo
                        if( abs(uvw.x)<0.66 )
                        {
                           float par = length(uvw.yz-float2(-1.05,0.65));
                           col *= lerp(float3(1,1,1),float3(0.6,0.2,0.8)*0.7,1.0-smoothstep(0.1,0.11,par));
                           col *= smoothstep(0.005,0.010,abs(par-0.105));
                        }                

                        // fake occlusion
                        focc = lerp(1.0,
                        0.03+0.97*smoothstep(-0.15,1.7,uvw.z),
                        smoothstep(-1.6,-1.3,uvw.y)*(1.0-clamp(dot(nor.xz,normalize(uvw.xz)),0.0,1.0))
                        );
                        
                        //frsha = lerp(1.0,
                        //            clamp(dot(nor.xz,normalize(uvw.xz)),0.0,1.0),
                        //            smoothstep(-1.6,-1.3,uvw.y)
                        //           );
                        //frsha *= smoothstep(0.85,1.0,length(uvw-float3(0.0,-1.0,0.0)));
                     }
                     else if( tm.y<4.5 )// hair
                     {
                        sss = 0.0;
                        col = (sin(cma.x)>0.7) ? float3(0.03,0.01,0.05)*1.5 :
                        float3(0.04,0.02,0.01)*0.4;
                        ks *= 0.75 + cma.z*18.0;
                        //float te = texture( iChannel2,float2( 0.25*atan(uvw.x,uvw.y),8.0*uvw.z) ).x;
                        float te = 0;
                        col *= 2.0*te;
                        ks *= 1.5*te;
                        
                        // fake occlusion
                        focc  = 1.0-smoothstep(-0.40, 0.8, uvw.y);
                        focc *= 1.0-0.95*smoothstep(-1.20,-0.2,-uvw.z);
                        focc *= 0.5+cma.z*12.0;
                        //frsha = 1.0-smoothstep(-1.3,-0.8,uvw.y);
                        //frsha *= 1.0-smoothstep(-1.20,-0.2,-uvw.z);
                     }
                     else if( tm.y<5.5 )// teeth
                     {
                        sss = 1.0;
                        col = float3(0.3,0.3,0.3);
                        ks *= 1.5;
                        //frsha = 0.0;
                     }

                     float fre = clamp(1.0+dot(nor,rd),0.0,1.0);
                     float occ = focc*calcOcclusion( pos, nor, time );

                     // --------------------------
                     // lighting. just four lights
                     // --------------------------
                     float3 lin = float3(0,0,0);

                     // fake sss
                     float nma = 0.0;
                     if( tm.y<1.5 )
                     {
                        nma = 1.0-smoothstep(0.0,0.12,length((uvw.xy-float2(0.0,-0.37))/float2(2.4,0.7)));
                     }

                     //float3 lig = normalize(float3(0.5,0.4,0.6));
                     float3 lig = float3(0.57,0.46,0.68);
                     float3 hal = normalize(lig-rd);
                     float dif = clamp( dot(nor,lig), 0.0, 1.0 );
                     //float sha = 0.0; if( dif>0.001 ) sha=calcSoftshadow( pos+nor*0.002, lig, 0.0001, 2.0, time, 5.0 );
                     float sha = calcSoftshadow( pos+nor*0.002, lig, 0.0001, 2.0, time, 5.0 );
                     float spe = 2.0*ks*pow(clamp(dot(nor,hal),0.0,1.0),se)*dif*sha*(0.04+0.96*pow(clamp(1.0-dot(hal,-rd),0.0,1.0),5.0));

                     // fake sss for key light
                     float3 cocc = lerp(float3(occ,occ,occ),
                     float3(0.1+0.9*occ,0.9*occ+0.1*occ*occ,0.8*occ+0.2*occ*occ),
                     tinterShadow);
                     cocc = lerp( cocc, float3(1,0.3,0.0), nma);
                     sha = lerp(sha,max(sha,0.3),nma);

                     float3  amb = cocc*(0.55 + 0.45*nor.y);
                     float bou = clamp(0.3-0.7*nor.x, 0.0, 1.0 );

                     lin +=      float3(0.65,1.05,2.0)*amb*1.15;
                     lin += 1.50*float3(1.60,1.40,1.2)*sdif(dot(nor,lig),0.5+0.3*nma+0.2*(1.0-occ)*tinterShadow) * lerp(float3(sha,sha,sha),float3(sha,0.2*sha+0.7*sha*sha,0.2*sha+0.7*sha*sha),tinterShadow);
                     lin +=      float3(1.00,0.30,0.1)*sss*fre*0.6*(0.5+0.5*dif*sha*amb)*(0.1+0.9*focc);
                     lin += 0.35*float3(4.00,2.00,1.0)*bou*occ*col;

                     col = lin*col + spe + fre*fre*fre*0.1*occ;

                     // overall
                     col *= 1.1;
                  }

                  //if( tm.x==-1.0) col=float3(1,0,0);
                  
                  return col;
               }

               // Animates the head turn. This is my first time animating
               // and I am aware I'm in uncanny/animatronic land. But I
               // have to start somwhere!
               //
               float animTurn( in float time )
               {	
                  const float w = 6.1;
                  float t = mod(time,w*2.0);
                  
                  float3 p = (t<w) ? float3(0.0,0.0,1.0) : float3(w,1.0,-1.0);
                  return p.y + p.z*expSustainedImpulse(t-p.x,1.0,10.0);
               }

               // Animates the eye blinks. Blinks are motivated by head
               // turns (again, to prevent animatronic and zoombie uncanny
               // valley stuff), but also there are random blinks. This
               // same funcion is called with some delay and extra
               // smmoothness to get the blink of the eyes be followed by
               // the face muscles around the face to react.
               //
               float animBlink( in float time, in float smo )
               {
                  // head-turn motivated blink
                  const float w = 6.1;
                  float t = mod(time-0.31,w*1.0);
                  float blink = smoothstep(0.0,0.1,t) - smoothstep(0.18,0.4,t);

                  // regular blink
                  float tt = mod(1.0+time,3.0);
                  blink = max(blink,smoothstep(0.0,0.07+0.07*smo,tt)-smoothstep(0.1+0.04*smo,0.35+0.3*smo,tt));
                  
                  // keep that eye alive always
                  float blinkBase = 0.04*(0.5+0.5*sin(time));
                  blink = lerp( blinkBase, 1.0, blink );

                  // base pose is a bit down
                  float down = 0.15;
                  return down+(1.0-down)*blink;
               }

               
               // The main rendering entry point. Basically it does some
               // setup or creating the ray that will explore the 3D scene
               // in search of the girl for each pixel, computes the
               // animation variables (blink, mouth and head movements),
               // does the rendering of the girl if it finds her, and
               // finally does gamme correction and some minimal color
               // processing and vignetting to the image.
               //
               half4 LitPassFragment(Varyings input) : SV_Target  {
                  half4 fragColor = half4 (1 , 1 , 1 , 1);
                  float2 fragCoord = ((input.screenPos.xy) / (input.screenPos.w + FLT_MIN)) * _ScreenParams.xy;
                  float3 tot = float3 (0.0 , 0.0 , 0.0);
                  #if AA > 1 
                     for (int m = ZEROExtended; m < AA; m++)
                     for (int n = ZEROExtended; n < AA; n++)
                     {
                        // pixel coordinates 
                        float2 o = float2 (float(m) , float(n)) / float(AA) - 0.5;
                        float2 p = (-_ScreenParams.xy + 2.0 * (fragCoord + o)) / _ScreenParams.y;
                        // time coordinate ( motion blurred , shutter = 0.5 ) 
                        float d = 0.5 * sin(fragCoord.x * 147.0) * sin(fragCoord.y * 131.0);
                        float time = _Time.y - 0.5 * (1.0 / 24.0) * (float(m * AA + n) + d) / float(AA * AA - 1);
                     #else 
                        float2 p = (-_ScreenParams.xy + 2.0 * fragCoord) / _ScreenParams.y;
                        float time = _Time.y;
                     #endif 
                     time += 2.0;

                     // camera movement	
                     float3 ro; float fl;
                     float3x3 ca = calcCamera( time, ro, fl );
                     float3 rd = mul(ca,normalize(float3((p-float2(-0.52,0.12))/1.1,fl)));


                     // animation (blink, face follow up, mouth)
                     float turn = animTurn( time );
                     animData.x = animBlink(time,0.0);
                     animData.y = animBlink(time-0.02,1.0);
                     animData.z = -0.25 + 0.2*(1.0-turn)*smoothstep(-0.3,0.9,sin(time*1.1)) + 0.05*cos(time*2.7);

                     // animation (head orientation)
                     animHead = float3( sin(time*0.5), sin(time*0.3), -cos(time*0.2) );
                     animHead = animHead*animHead*animHead;
                     animHead.x = -0.025*animHead.x + 0.2*(0.7+0.3*turn);
                     animHead.y =  0.1 + 0.02*animHead.y*animHead.y*animHead.y;
                     animHead.z = -0.03*(0.5 + 0.5*animHead.z) - (1.0-turn)*0.05;

                     float3 col = 0;

                     float tmin = 0;

                     if( p.x*1.4+p.y<0.8 && -p.x*4.5+p.y<6.5 && p.x<0.48)
                     col = renderGirl(p,ro,rd,tmin,col,time);
                   
                     // gamma 
                     col = pow(abs(col) , float3 (0.4545, 0.4545, 0.4545));
                     tot += col;
                     #if AA > 1 
                     }
                     tot /= float(AA * AA);
                  #endif 

                  // compress
                  tot = 3.8*tot/(3.0+dot(tot,float3(0.333,0.333,0.333)));

                  // vignetting 
                  float2 q = fragCoord / _ScreenParams.xy;
                  tot *= 0.5 + 0.5 * pow(abs(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y)) , 0.15);

                  // grade
                  tot = tot*float3(1.02,1.00,0.99)+float3(0.0,0.0,0.045);

                  // output 
                  fragColor = float4 (tot , 1.0);
                  return fragColor - 0.1;
               }
               ENDHLSL
            }
         }
      }