'use strict';
(() => {
const D=window.WEEK7_DATA,$=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const C={ink:'#263740',muted:'#626d69',teal:'#087f79',blue:'#365d93',rust:'#bb5c38',gold:'#bf914d',paper:'#fffdf8',line:'#d4d9d0'};
const slides=[...document.querySelectorAll('.slide')],fmt=(x,n=2)=>(Math.abs(x)<.5*10**(-n)?0:x).toFixed(n),pair=p=>'['+p.map(x=>fmt(x)).join(', ')+']';
const sources={math:['PCA geometry reference','https://scikit-learn.org/stable/modules/decomposition.html#pca','Instructor reference example'],kather:['Kather et al. (2016) · CC BY 4.0','https://doi.org/10.5281/zenodo.53169','Reference calculation · training basis · validation tiles'],wdbc:['UCI · Breast Cancer Wisconsin (Diagnostic)','https://doi.org/10.24432/C5DW2B','Reference calculation · Week 6 training split'],course:['Week 7 implementation checkpoint','#slide-11','Student worksheet contains commands and submission details']};
slides.forEach((s,i)=>{const [label,url,note]=sources[s.dataset.source],left=document.createElement('span'),right=document.createElement('span'),a=document.createElement('a');a.textContent=label;a.href=url;if(!url.startsWith('#')){a.target='_blank';a.rel='noopener'}left.append(a);right.textContent=note;s.querySelector('.source').append(left,right);const option=document.createElement('option');option.value=i;option.textContent=String(i+1).padStart(2,'0')+' · '+s.dataset.title;$('slide-picker').append(option)});
let page=0;
function show(i,hash=true){page=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,j)=>{s.classList.toggle('active',j===page);s.setAttribute('aria-hidden',j!==page)});$('slide-picker').value=page;$('count').textContent=`${page+1} / ${slides.length}`;$('previous').disabled=page===0;$('next').disabled=page===slides.length-1;$('progress-fill').style.width=(page+1)/slides.length*100+'%';if(hash)history.replaceState(null,'','#slide-'+(page+1));window.scrollTo(0,0);$('note-text').textContent=slides[page].dataset.notes;document.title=slides[page].dataset.title+' · Python Workshop 07'}
function fromHash(){const m=location.hash.match(/^#slide-(\d+)$/);show(m?+m[1]-1:0,false)}
$('next').onclick=()=>show(page+1);$('previous').onclick=()=>show(page-1);$('slide-picker').onchange=e=>show(+e.target.value);window.addEventListener('hashchange',fromHash);
$('notes-button').onclick=()=>{$('note-text').textContent=slides[page].dataset.notes;$('notes-title').textContent=`${page+1}. ${slides[page].dataset.title}`;$('notes').showModal()};$('close-notes').onclick=()=>$('notes').close();$('notes').addEventListener('click',e=>{if(e.target===$('notes'))$('notes').close()});
$('fullscreen-button').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen()}catch{$('fullscreen-button').textContent='Fullscreen unavailable'}};$('print-button').onclick=()=>window.print();
document.addEventListener('keydown',e=>{if($('notes').open||e.altKey||e.ctrlKey||e.metaKey||e.target.closest('input,select,textarea,button,summary'))return;if(e.key==='ArrowRight'||e.key==='PageDown'){e.preventDefault();show(page+1)}if(e.key==='ArrowLeft'||e.key==='PageUp'){e.preventDefault();show(page-1)}if(e.key==='Home'){e.preventDefault();show(0)}if(e.key==='End'){e.preventDefault();show(slides.length-1)}if(e.key.toLowerCase()==='n')$('notes-button').click()});
const add=(svg,tag,attrs={},str)=>{const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(str!==undefined)n.textContent=str;svg.append(n);return n};
const line=(s,x1,y1,x2,y2,color=C.line,width=1,dash)=>add(s,'line',{x1,y1,x2,y2,stroke:color,'stroke-width':width,...(dash?{'stroke-dasharray':dash}:{})});
const text=(s,x,y,t,anchor='start',color=C.ink,size=14)=>add(s,'text',{x,y,'text-anchor':anchor,style:`fill:${color};font-size:${size}px`},t);
const circle=(s,x,y,r,color,filled=true)=>add(s,'circle',{cx:x,cy:y,r,fill:filled?color:C.paper,stroke:color,'stroke-width':2});
function path(s,points,color,width=2,dash){return add(s,'path',{d:points.map((p,i)=>(i?'L':'M')+p.join(',')).join(' '),fill:'none',stroke:color,'stroke-width':width,...(dash?{'stroke-dasharray':dash}:{})})}
function scatter(id,points,{range=[-4,4],projected=null,selected=-1,mean=[0,0],directions=[[1,0]],labels=['Feature 1','Feature 2'],allSelected=false}={}){
 const s=$(id);s.replaceChildren();const [lo,hi]=range,scale=260/(hi-lo),sx=x=>140+(x-lo)*scale,sy=y=>294-(y-lo)*scale;
 for(let v=Math.ceil(lo);v<=hi;v++){if(hi-lo>10&&v%2)continue;line(s,sx(lo),sy(v),sx(hi),sy(v));line(s,sx(v),sy(lo),sx(v),sy(hi));text(s,sx(v),313,v,'middle',C.muted,11);text(s,131,sy(v)+4,v,'end',C.muted,11)}
 text(s,270,340,labels[0],'middle',C.ink,14);const yl=text(s,93,164,labels[1],'middle',C.ink,14);yl.setAttribute('transform','rotate(-90 93 164)');
 const clipId=id+'-clip';const defs=add(s,'defs'),clip=add(defs,'clipPath',{id:clipId});add(clip,'rect',{x:sx(lo),y:sy(hi),width:260,height:260});const plot=add(s,'g',{'clip-path':`url(#${clipId})`});
 directions.forEach((v,j)=>{const length=hi-lo;line(plot,sx(mean[0]-length*v[0]),sy(mean[1]-length*v[1]),sx(mean[0]+length*v[0]),sy(mean[1]+length*v[1]),j?C.gold:C.blue,2)});
 if(projected)points.forEach((p,i)=>{line(plot,sx(p[0]),sy(p[1]),sx(projected[i][0]),sy(projected[i][1]),i===selected?C.rust:C.muted,points.length>20?.5:1.4,'4 3');circle(plot,sx(projected[i][0]),sy(projected[i][1]),points.length>20?2.5:6,C.blue,false)});
 points.forEach((p,i)=>{const c=i===selected?C.rust:C.teal,n=circle(plot,sx(p[0]),sy(p[1]),points.length>20?2.2:6,c);add(n,'title',{},`Observation ${i+1}: ${pair(p)}`)});
 line(plot,sx(mean[0])-6,sy(mean[1]),sx(mean[0])+6,sy(mean[1]),C.ink,2);line(plot,sx(mean[0]),sy(mean[1])-6,sx(mean[0]),sy(mean[1])+6,C.ink,2);
 text(s,148,19,'+ mean','start',C.ink,12);text(s,252,19,directions.length===1?'Blue: direction 1':'Blue: direction 1   Gold: direction 2','start',C.muted,11);
 if(allSelected)points.forEach((p,i)=>text(s,sx(p[0])+9,sy(p[1])-9,`z = ${p[0]}`,'start',C.teal,12));
}
const points=[[2,3],[-2,1],[1,-2]];
scatter('shared-chart',points,{allSelected:true});
points.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=pair(p);$('project-sample').append(o)});
function projectExample(){const i=+$('project-sample').value,p=points[i];scatter('project-chart',points,{selected:i,projected:points.map(p=>[p[0],0])});$('project-readout').textContent=`z₁ = ${p[0]} × 1 + ${p[1]} × 0 = ${p[0]}`}
$('project-sample').onchange=projectExample;
function shiftedExample(shift){document.querySelectorAll('[data-mean]').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.mean===shift));const m=shift?[4,2]:[0,0],p=[m[0]+2,m[1]+3],q=[m[0]+2,m[1]];scatter('reconstruct-chart',[p],{mean:m,projected:[q],selected:0,range:shift?[-1,7]:[-4,4]});$('reconstruct-readout').textContent=`x = ${pair(p)}. Score z₁ = 2. Reconstruction x̂ = ${pair(q)}.`}
document.querySelectorAll('[data-mean]').forEach(b=>b.onclick=()=>shiftedExample(+b.dataset.mean));
function keep(k){document.querySelectorAll('[data-keep]').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.keep===k));const m=[4,2],x=points.map(p=>p.map((v,j)=>v+m[j])),r=points.map(p=>[p[0]+4,k===1?2:p[1]+2]),mse=x.reduce((s,p,i)=>s+p.reduce((a,v,j)=>a+(v-r[i][j])**2,0),0)/6;scatter('keep-chart',x,{mean:m,projected:r,range:[-1,7],directions:k===1?[[1,0]]:[[1,0],[0,1]]});$('keep-readout').textContent=`${k} score${k>1?'s':''} per observation. MSE = ${fmt(mse,4)}. ${k===1?'Vertical residuals remain.':'Every point returns to its input location.'}`}
document.querySelectorAll('[data-keep]').forEach(b=>b.onclick=()=>keep(+b.dataset.keep));
function wdbc(k){document.querySelectorAll('[data-wdbc]').forEach(b=>b.setAttribute('aria-pressed',+b.dataset.wdbc===k));const w=D.wdbc;scatter('wdbc-chart',w.X,{range:w.range,mean:w.mean,projected:w.reconstructed[k],directions:w.components.slice(0,k),labels:['Mean radius (standardized)','Mean texture (standardized)']});$('wdbc-readout').textContent=`${k} PC${k>1?'s':''}. Reconstruction MSE = ${fmt(w.mse[k],6)} in standardized coordinates.`}
document.querySelectorAll('[data-wdbc]').forEach(b=>b.onclick=()=>wdbc(+b.dataset.wdbc));
function mseChart(j,k){const s=$('mse-chart');s.replaceChildren();const t=D.tissue,left=74,right=448,top=30,bottom=235,max=Math.max(...t.mean_mse,...t.sample_mse[j])*1.12,sx=x=>left+Math.log10(x/3)/Math.log10(100)*(right-left),sy=y=>bottom-y/max*(bottom-top);line(s,left,top,left,bottom,C.muted);line(s,left,bottom,right,bottom,C.muted);
 for(let q=0;q<=4;q++){const y=max*q/4;line(s,left,sy(y),right,sy(y));text(s,left-8,sy(y)+4,fmt(y,3),'end',C.muted,12)}
 t.k.forEach(v=>{line(s,sx(v),top,sx(v),bottom,C.line,1);text(s,sx(v),255,v,'middle',C.muted,12)});
 path(s,t.k.map((v,i)=>[sx(v),sy(t.mean_mse[i])]),C.teal,2.5);path(s,t.k.map((v,i)=>[sx(v),sy(t.sample_mse[j][i])]),C.rust,2,'6 3');t.k.forEach((v,i)=>{circle(s,sx(v),sy(t.mean_mse[i]),i===k?5:3,C.teal);circle(s,sx(v),sy(t.sample_mse[j][i]),i===k?5:3,C.rust)});
 text(s,74,17,'MSE','start',C.ink,13);text(s,261,285,'Number of components (log spacing)','middle',C.ink,13);text(s,170,17,'Eight-image mean','start',C.teal,12);text(s,325,17,'Selected tile','start',C.rust,12);
}
D.tissue.classes.forEach((name,j)=>{const o=document.createElement('option');o.value=j;o.textContent=`${name} · #${D.tissue.ids[j]}`;$('tissue-sample').append(o)});
function tissue(){const j=+$('tissue-sample').value,index=+$('tissue-k').value,k=D.tissue.k[index],t=D.tissue;$('tissue-k-value').textContent=k;$('tissue-original').src=`assets/tissue-source-${j}.png`;$('tissue-original').alt=`Original ${t.classes[j]} tissue tile ${t.ids[j]}, 150 by 150`;$('tissue-input').src=`assets/tissue-input-${j}.png`;$('tissue-input').alt=`Model input for ${t.classes[j]} tile ${t.ids[j]}, 28 by 28 RGB`;$('tissue-reconstruction').src=`assets/tissue-k${k}-${j}.png`;$('tissue-reconstruction').alt=`${t.classes[j]} tile reconstructed with ${k} components`;$('tissue-recon-caption').textContent=`${k} PCs · 28 × 28`;$('tissue-readout').textContent=`This tile’s MSE: ${fmt(t.sample_mse[j][index],5)}. Eight-image mean: ${fmt(t.mean_mse[index],5)}. ${k} scores per image, plus the shared basis.`;mseChart(j,index)}
$('tissue-sample').onchange=tissue;$('tissue-k').oninput=tissue;
// Hidden slides are initialized too so printing has all figures.
projectExample();shiftedExample(0);keep(1);wdbc(1);tissue();fromHash();
})();
