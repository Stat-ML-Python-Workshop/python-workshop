/* Offline three-feature variance-bound illustration. No training or network calls. */
(function(){
  'use strict';
  const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
  const scale=(a,s)=>a.map(x=>x*s),add=(a,b)=>a.map((x,i)=>x+b[i]);
  const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const norm=a=>Math.hypot(...a),unit=a=>scale(a,1/norm(a));
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  function rotate(v,axis,angle){const c=Math.cos(angle),s=Math.sin(angle);return add(add(scale(v,c),scale(cross(axis,v),s)),scale(axis,dot(axis,v)*(1-c)));}
  window.createVarianceExplorer=function(root,data){
    if(!root||root.varianceExplorer)return root&&root.varianceExplorer;
    const prefix=root.id+'-';
    const q=data.components,eigenvalues=data.eigenvalues,points=data.points;
    const initial=unit(add(add(scale(q[0],.68),scale(q[1],-.57)),scale(q[2],.46)));
    let v=initial,yaw=.7854,pitch=.32,mode='direction',drag=null,animation=0,frame=0;
    const colors={v:'',pc:[],ink:'',muted:'',line:'',paper:''};
    root.innerHTML=`<div class="vp-layout">
      <div class="vp-scene"><canvas role="img" aria-label="Three-dimensional synthetic sample cloud, principal axes, and movable unit direction v. Use the direction sliders or maximum and minimum buttons for keyboard access."></canvas>
        <div class="vp-caption">240 centered samples · 3 features · direction arrows enlarged</div>
        <div class="vp-unit vp-numbers"></div>
      </div>
      <div class="vp-panel">
        <div class="vp-actions"><button type="button" data-vp="max">Maximum · PC1</button><button type="button" data-vp="min">Minimum · PC3</button></div>
        <label class="vp-drag" for="${prefix}mode">Drag to <select id="${prefix}mode" data-vp="mode"><option value="direction">move direction v</option><option value="camera">rotate the view</option></select></label>
        <label for="${prefix}azimuth"><span class="vp-control-head"><span>Direction azimuth</span><output data-vp="azimuth-value" for="${prefix}azimuth"></output></span><input id="${prefix}azimuth" data-vp="azimuth" type="range" min="-180" max="180" step="0.1"></label>
        <label for="${prefix}elevation"><span class="vp-control-head"><span>Direction elevation</span><output data-vp="elevation-value" for="${prefix}elevation"></output></span><input id="${prefix}elevation" data-vp="elevation" type="range" min="-90" max="90" step="0.1"></label>
        <div><div class="vp-section-label">PC direction weights aᵢ²</div><div class="vp-weights">${q.map((_,i)=>`<div><div class="vp-weight-head"><span class="vp-pc-label"><span class="vp-key" style="background:var(--vp-pc${i+1})"></span>PC${i+1} · λ${['₁','₂','₃'][i]} = ${eigenvalues[i].toFixed(2)}</span><output data-vp="weight-${i}"></output></div><div class="vp-weight-track" role="meter" aria-label="PC${i+1} squared direction weight" aria-valuemin="0" aria-valuemax="100" data-vp="meter-${i}"><div class="vp-weight-fill" data-vp="fill-${i}" style="background:var(--vp-pc${i+1})"></div></div></div>`).join('')}</div></div>
        <div class="vp-coefficients"></div>
        <div class="vp-variance"><div class="vp-control-head"><span>Projection variance</span><output class="vp-variance-value" data-vp="variance"></output></div><div class="vp-bound-track"><div class="vp-bound-fill"></div><div class="vp-bound-marker"></div></div><div class="vp-bound-labels"><span>λ₃ = ${eigenvalues[2].toFixed(2)}</span><span>λ₁ = ${eigenvalues[0].toFixed(2)}</span></div></div>
        <div class="vp-state" role="status" aria-live="polite"></div>
      </div></div>`;
    const el=name=>root.querySelector(`[data-vp="${name}"]`);
    const canvas=root.querySelector('canvas'),ctx=canvas.getContext('2d');
    const span=document.createElement('span');span.hidden=true;root.append(span);
    const resolve=name=>{span.style.color=`var(${name})`;return getComputedStyle(span).color;};
    function readColors(){colors.v=resolve('--vp-v');colors.pc=[1,2,3].map(i=>resolve('--vp-pc'+i));colors.ink=resolve('--ink');colors.muted=resolve('--muted');colors.line=resolve('--line');colors.paper=resolve('--paper');requestDraw();}
    function snapshot(){const coefficients=q.map(axis=>dot(axis,v));const weights=coefficients.map(x=>x*x);const scores=points.map(x=>dot(x,v));const mean=scores.reduce((a,b)=>a+b,0)/scores.length;const variance=scores.reduce((s,z)=>s+(z-mean)**2,0)/(scores.length-1);return{v:v.slice(),coefficients,weights,variance,spectralVariance:dot(weights,eigenvalues),camera:{yaw,pitch},mode};}
    function update(announce=false){
      const state=snapshot();
      const azimuth=Math.atan2(v[1],v[0])*180/Math.PI,elevation=Math.asin(clamp(v[2],-1,1))*180/Math.PI;
      el('azimuth').value=azimuth;el('elevation').value=elevation;
      el('azimuth-value').value=azimuth.toFixed(1)+'°';el('elevation-value').value=elevation.toFixed(1)+'°';
      state.weights.forEach((w,i)=>{const pct=clamp(w,0,1)*100;el('weight-'+i).value=pct.toFixed(1)+'%';el('fill-'+i).style.width=pct+'%';el('meter-'+i).setAttribute('aria-valuenow',pct.toFixed(3));});
      root.querySelector('.vp-coefficients').textContent='Signed a: ['+state.coefficients.map(x=>(Math.abs(x)<.00005?0:x).toFixed(3)).join(', ')+']';
      el('variance').value=state.variance.toFixed(4);
      const fraction=clamp((state.variance-eigenvalues[2])/(eigenvalues[0]-eigenvalues[2]),0,1);
      root.querySelector('.vp-bound-fill').style.width=(fraction*100)+'%';root.querySelector('.vp-bound-marker').style.left=(fraction*100)+'%';
      root.querySelector('.vp-unit').textContent='v = ['+v.map(x=>(Math.abs(x)<.00005?0:x).toFixed(3)).join(', ')+'] · ‖v‖ = '+norm(v).toFixed(3);
      const extreme=state.weights.findIndex(x=>x>1-1e-10);
      el('max').setAttribute('aria-pressed',String(extreme===0));el('min').setAttribute('aria-pressed',String(extreme===2));
      if(announce)root.querySelector('.vp-state').textContent=extreme>=0?`PC${extreme+1}: variance ${state.variance.toFixed(4)}, weight 100%.`:`Variance ${state.variance.toFixed(4)}; weights ${state.weights.map(w=>(w*100).toFixed(1)+'%').join(', ')}.`;
      requestDraw();
    }
    function setDirection(value,announce=true){if(value.length!==3||!value.every(Number.isFinite)||norm(value)<1e-12)throw new Error('A finite nonzero three-dimensional direction is required.');cancelAnimationFrame(animation);v=unit(value);update(announce);}
    function save(){const state=snapshot();try{const pending=window.openai?.setWidgetState?.({modelContent:{direction:state.v,weights:state.weights,variance:state.variance},privateContent:{camera:state.camera,mode}});pending?.catch(()=>{});}catch(_){}}
    function snap(index){
      cancelAnimationFrame(animation);const start=v.slice(),target=unit(q[index]),cos=clamp(dot(start,target),-1,1),angle=Math.acos(cos),begin=performance.now();
      if(matchMedia('(prefers-reduced-motion: reduce)').matches){setDirection(target);save();return;}
      let tangent=add(target,scale(start,-cos));if(norm(tangent)<1e-7)tangent=cross(start,Math.abs(start[0])<.8?[1,0,0]:[0,1,0]);tangent=unit(tangent);
      const tick=now=>{const t=clamp((now-begin)/320,0,1),e=t*t*(3-2*t);v=t===1?target:unit(add(scale(start,Math.cos(angle*e)),scale(tangent,Math.sin(angle*e))));update(t===1);if(t<1)animation=requestAnimationFrame(tick);else save();};
      animation=requestAnimationFrame(tick);
    }
    function basis(){return{right:[Math.cos(yaw),Math.sin(yaw),0],up:[-Math.sin(yaw)*Math.sin(pitch),Math.cos(yaw)*Math.sin(pitch),Math.cos(pitch)],depth:[Math.sin(yaw)*Math.cos(pitch),-Math.cos(yaw)*Math.cos(pitch),Math.sin(pitch)]};}
    let width=0,height=0;
    const radius=Math.max(...points.map(norm),6.5)*1.13;
    function requestDraw(){cancelAnimationFrame(frame);frame=requestAnimationFrame(draw);}
    function draw(){
      const rect=canvas.getBoundingClientRect();if(rect.width<2||rect.height<2)return;
      width=rect.width;height=rect.height;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
      const b=basis(),length=radius*.60,axisLength=radius*.70;
      const extentX=Math.max(axisLength,...points.map(p=>Math.abs(dot(p,b.right))));
      const extentY=Math.max(axisLength,...points.map(p=>Math.abs(dot(p,b.up))));
      const s=Math.min((width-64)/(2*extentX),(height-52)/(2*extentY)),cx=width*.5,cy=height*.51;
      const project=p=>[cx+dot(p,b.right)*s,cy-dot(p,b.up)*s,dot(p,b.depth)];
      function segment(a,z,color,lineWidth=1,alpha=1,dashed=false){const p=project(a),r=project(z);ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.setLineDash(dashed?[5,4]:[]);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(r[0],r[1]);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;}
      function label(text,p,color,offset=9){const r=project(p);ctx.fillStyle=color;ctx.font='13px Arial, sans-serif';ctx.textAlign='center';ctx.fillText(text,clamp(r[0],30,width-30),clamp(r[1]-offset,18,height-9));}
      [[1,0,0],[0,1,0],[0,0,1]].forEach((a,i)=>{segment(scale(a,-axisLength),scale(a,axisLength),colors.line,1);label('X'+(i+1),scale(a,axisLength),colors.muted,5);});
      q.forEach((a,i)=>{segment(scale(a,-length),scale(a,length),colors.pc[i],1.5,.7,true);label('PC'+(i+1),scale(a,length),colors.pc[i],i===0?15:9);});
      const projected=points.map(p=>({p,screen:project(p)})).sort((a,z)=>a.screen[2]-z.screen[2]);
      projected.forEach(({p,screen},i)=>{ctx.globalAlpha=.30+.35*(i/points.length);ctx.fillStyle=colors.muted;ctx.beginPath();ctx.arc(screen[0],screen[1],2.5,0,2*Math.PI);ctx.fill();});ctx.globalAlpha=1;
      const arrow=scale(v,length);
      points.forEach((p,i)=>{const projectedPoint=scale(v,dot(p,v));if(i%8===0)segment(p,projectedPoint,colors.v,.7,.12);const r=project(projectedPoint);ctx.fillStyle=colors.v;ctx.globalAlpha=.42;ctx.beginPath();ctx.arc(r[0],r[1],1.8,0,2*Math.PI);ctx.fill();});ctx.globalAlpha=1;
      segment(scale(arrow,-1),arrow,colors.v,2.5);
      const end=project(arrow),origin=project([0,0,0]),angle2=Math.atan2(end[1]-origin[1],end[0]-origin[0]);
      ctx.fillStyle=colors.v;ctx.beginPath();ctx.moveTo(end[0],end[1]);ctx.lineTo(end[0]-12*Math.cos(angle2-.45),end[1]-12*Math.sin(angle2-.45));ctx.lineTo(end[0]-12*Math.cos(angle2+.45),end[1]-12*Math.sin(angle2+.45));ctx.closePath();ctx.fill();
      ctx.beginPath();ctx.arc(end[0],end[1],5.5,0,2*Math.PI);ctx.fill();label('v',arrow,colors.v,34);
      ctx.fillStyle=colors.ink;ctx.beginPath();ctx.arc(origin[0],origin[1],3,0,2*Math.PI);ctx.fill();
      ctx.font='12px Arial, sans-serif';ctx.textAlign='left';ctx.fillStyle=colors.muted;ctx.fillText(mode==='direction'?'Drag: move v':'Drag: rotate view',12,height-12);
    }
    el('max').addEventListener('click',()=>snap(0));el('min').addEventListener('click',()=>snap(2));
    el('mode').addEventListener('change',()=>{mode=el('mode').value;requestDraw();save();});
    function slider(){const az=+el('azimuth').value*Math.PI/180,elev=+el('elevation').value*Math.PI/180;setDirection([Math.cos(elev)*Math.cos(az),Math.cos(elev)*Math.sin(az),Math.sin(elev)]);}
    ['azimuth','elevation'].forEach(name=>{el(name).addEventListener('input',slider);el(name).addEventListener('change',save);});
    canvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;cancelAnimationFrame(animation);canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY};});
    canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};if(mode==='camera'){yaw-=dx*.008;pitch=clamp(pitch+dy*.008,-1.35,1.35);requestDraw();}else{const b=basis();v=unit(rotate(rotate(v,b.up,dx*.009),b.right,dy*.009));update();}});
    function finish(){if(!drag)return;drag=null;update(true);save();}
    canvas.addEventListener('pointerup',finish);canvas.addEventListener('pointercancel',finish);canvas.addEventListener('lostpointercapture',finish);
    function restore(saved){if(!saved)return;const mc=saved.modelContent,pc=saved.privateContent;if(mc?.direction?.length===3&&mc.direction.every(Number.isFinite)&&norm(mc.direction)>1e-12)v=unit(mc.direction);if(pc?.camera&&Number.isFinite(pc.camera.yaw)&&Number.isFinite(pc.camera.pitch)){yaw=pc.camera.yaw;pitch=clamp(pc.camera.pitch,-1.35,1.35);}if(['direction','camera'].includes(pc?.mode)){mode=pc.mode;el('mode').value=mode;}update();}
    const observer=new ResizeObserver(()=>requestDraw());observer.observe(canvas);
    new MutationObserver(readColors).observe(document.documentElement,{attributes:true,attributeFilter:['class','style','data-theme']});
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change',readColors);
    window.addEventListener('beforeprint',()=>{cancelAnimationFrame(animation);draw();});
    window.addEventListener('openai:set_globals',e=>restore(e.detail?.globals?.widgetState));
    const api={snapshot,setDirection,refresh:()=>{readColors();update();},data};root.varianceExplorer=api;
    readColors();update();restore(window.openai?.widgetState);return api;
  };
})();
