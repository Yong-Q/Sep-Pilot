"""Readonly server instrumentation, executed only inside the OS plot sandbox.

The Python interpreter is shared with generated code. Observations are advisory,
not tamper-proof scientific facts or success/authorization. The native parent
still independently validates isolation, exit status and PNG bytes; independent
visual review remains required. No network, process or host-file access is added.
"""
import json
import math
import sys

PREFIX = 'SEPPILOT_RENDER_DIAGNOSTIC '
MAX_POINTS = 200000
MAX_AXES = 32
MAX_ARTISTS = 128


class NativePlotVisibilityError(ValueError):
    pass


def _numbers(values):
    result=[]
    for v in values:
        try:v=float(v)
        except (ValueError,TypeError,OverflowError):v=None
        result.append(v if v is not None and math.isfinite(v)else None)
    return result


def inspect_figure(figure):
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.transforms import Bbox
    from matplotlib.text import Text
    figure.canvas.draw()
    reports=[];blocking=[];notes=[];artist_total=0;point_total=0
    for index,ax in enumerate(figure.axes[:MAX_AXES]):
        report={'axis':index,'title':str(ax.get_title())[:200],
            'x_label':str(ax.get_xlabel())[:200],'y_label':str(ax.get_ylabel())[:200],
            'dataLim':_numbers(ax.dataLim.bounds),'viewLim':_numbers(ax.viewLim.bounds),
            'x_limits':_numbers(ax.get_xlim()),'y_limits':_numbers(ax.get_ylim()),
            'x_scale':ax.get_xscale(),'y_scale':ax.get_yscale(),
            'autoscale_x':bool(ax.get_autoscalex_on()),'autoscale_y':bool(ax.get_autoscaley_on()),
            'axis_visible':bool(ax.axison and ax.get_visible()),'artists':[]}
        categorical=any(getattr(getattr(axis,'converter',None),'__class__',type(None)).__module__=='matplotlib.category'for axis in(ax.xaxis,ax.yaxis))
        report['categorical_units']=categorical
        for kind,artists in(('line',ax.lines),('scatter',[a for a in ax.collections if isinstance(a,PathCollection)])):
            for artist_index,artist in enumerate(artists[:256]):
                if not artist.get_visible():continue
                artist_total+=1
                if artist_total>MAX_ARTISTS:notes.append('artist audit bound reached; remaining geometry unreviewed');break
                transform=artist.get_transform()if kind=='line'else artist.get_offset_transform()
                if transform is not ax.transData:
                    continue # axes-coordinate/reference/diagram geometry is not observational data
                try:
                    if kind=='line':
                        x=artist.get_xdata(orig=False);y=artist.get_ydata(orig=False)
                        size=max(len(x),len(y))
                        if size>MAX_POINTS:notes.append('axis '+str(index)+' point bound exceeded; not a complete geometry audit');continue
                        values=np.ma.column_stack((x,y))
                    else:
                        values=artist.get_offsets();size=len(values)
                        if size>MAX_POINTS:notes.append('axis '+str(index)+' point bound exceeded; not a complete geometry audit');continue
                    point_total+=size
                    if point_total>MAX_POINTS:notes.append('figure point budget reached; remaining geometry unreviewed');continue
                    values=np.ma.asarray(values,dtype=float)
                    if values.ndim!=2 or values.shape[1]!=2:continue
                    raw=np.asarray(values.filled(np.nan))
                    finite=np.isfinite(raw).all(axis=1)
                    domain=finite.copy()
                    if ax.get_xscale()=='log':domain&=raw[:,0]>0
                    if ax.get_yscale()=='log':domain&=raw[:,1]>0
                    observations=raw[domain]
                    item={'kind':kind,'artist':artist_index,'finite_points':int(finite.sum()),
                        'domain_points':len(observations),'visible_points':0,'visible_segment':False}
                    if not len(observations):report['artists'].append(item);continue
                    fractions=ax.transAxes.inverted().transform(ax.transData.transform(observations))
                    valid=np.isfinite(fractions).all(axis=1)
                    inside=valid&(fractions[:,0]>=-1e-9)&(fractions[:,0]<=1+1e-9)&(fractions[:,1]>=-1e-9)&(fractions[:,1]<=1+1e-9)
                    item['visible_points']=int(inside.sum())
                    item['data_x_range']=[float(observations[:,0].min()),float(observations[:,0].max())]
                    item['data_y_range']=[float(observations[:,1].min()),float(observations[:,1].max())]
                    if kind=='line'and artist.get_linestyle()not in('None','none','',' '):
                        # Endpoints may both be outside a normal zoom while the
                        # actual rendered line segment crosses the viewport.
                        path=artist.get_path().transformed(artist.get_transform())
                        item['visible_segment']=bool(path.intersects_bbox(ax.bbox,filled=False))
                    clipped=not item['visible_points']and not item['visible_segment']
                    item['all_observations_outside_view']=clipped
                    explicit=not report['autoscale_x']or not report['autoscale_y']
                    quantitative=report['axis_visible']and bool(report['x_label']or report['y_label'])and not categorical
                    # Whole-axis empty ROI zooms are ambiguous. Fail only the
                    # specific orthogonal-clipping case: one data coordinate is
                    # fully in view, the other entirely outside explicit limits.
                    xin=valid&(fractions[:,0]>=-1e-9)&(fractions[:,0]<=1+1e-9)
                    yin=valid&(fractions[:,1]>=-1e-9)&(fractions[:,1]<=1+1e-9)
                    x_disjoint=bool(valid.all()and((fractions[:,0]<0).all()or(fractions[:,0]>1).all()))
                    y_disjoint=bool(valid.all()and((fractions[:,1]<0).all()or(fractions[:,1]>1).all()))
                    orthogonal=bool((xin.all()and y_disjoint)or(yin.all()and x_disjoint))
                    if clipped and explicit and quantitative and orthogonal:
                        message=('axis '+str(index)+' '+kind+' '+str(artist_index)+': all '+str(len(observations))+
                            ' finite data observations are outside explicit view limits; visible_points=0, visible_segment=false; '+
                            'data_x='+str(item['data_x_range'])+', data_y='+str(item['data_y_range'])+
                            ', x_limits='+str(report['x_limits'])+', y_limits='+str(report['y_limits'])+
                            '. Correct the data-coordinate axis limits or explicitly redesign the panel; never substitute stage/category ticks for numerical quantity values.')
                        blocking.append(message[:1000])
                    elif clipped:notes.append('axis '+str(index)+' '+kind+' '+str(artist_index)+': data outside view; ambiguous ROI/diagram/category intent, advisory only')
                    report['artists'].append(item)
                except (ValueError,TypeError,OverflowError,AttributeError)as error:
                    notes.append('axis '+str(index)+' '+kind+': geometry unavailable '+type(error).__name__)
        reports.append(report)
    # Layout boxes are notes only. A bbox_inches=tight export can legitimately
    # include text outside the pre-export figure extent.
    try:
        renderer=figure.canvas.get_renderer();texts=[]
        for ax in figure.axes[:MAX_AXES]:texts.extend([ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts[:128]])
        texts.extend(figure.texts[:128]);boxes=[]
        for text in texts[:512]:
            if isinstance(text,Text)and text.get_visible()and text.get_text().strip():
                box=text.get_window_extent(renderer)
                if np.isfinite(box.extents).all():boxes.append((text.get_text()[:80],box))
        overlaps=[]
        for i,(a,box)in enumerate(boxes):
            for b,other in boxes[i+1:]:
                intersection=Bbox.intersection(box,other)
                if intersection is not None and intersection.width>2 and intersection.height>2:
                    overlaps.append({'texts':[a,b],'overlap_pixels':_numbers(intersection.bounds)})
                    if len(overlaps)>=16:break
            if len(overlaps)>=16:break
        if overlaps:notes.append({'layout_bbox_overlaps':overlaps,'scope':'pre-export text extents; advisory only, may include intentional overlap'})
    except (ValueError,TypeError,AttributeError):notes.append('layout bbox audit unavailable')
    return {'schema_version':1,'evidence_class':'in_process_runtime_advisory',
        'independently_verified':False,'scientific_acceptance':False,
        'axes':reports,'blocking_diagnostics':blocking[:16],'notes':notes[:32]}


def install():
    from matplotlib.figure import Figure
    original=Figure.savefig
    def checked_savefig(self,*args,**kwargs):
        report=inspect_figure(self)
        encoded=json.dumps(report,ensure_ascii=False,allow_nan=False)
        if len(encoded.encode())>32768:
            report['notes']=report['notes'][:8]+['Diagnostic projection truncated; this is not a complete geometry audit.']
            report['axes']=report['axes'][:16]
            for axis in report['axes']:axis['artists']=axis['artists'][:8]
            report['blocking_diagnostics']=report['blocking_diagnostics'][:4]
            encoded=json.dumps(report,ensure_ascii=False,allow_nan=False)
        sys.stderr.write(PREFIX+encoded+'\n');sys.stderr.flush()
        if report['blocking_diagnostics']:
            raise NativePlotVisibilityError('\n'.join(report['blocking_diagnostics'][:4]))
        return original(self,*args,**kwargs)
    Figure.savefig=checked_savefig
