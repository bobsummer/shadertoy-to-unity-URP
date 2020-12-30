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