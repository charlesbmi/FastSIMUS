"""Plot helpers for the wavefield explorer.

Kept next to the notebook rather than in FastSIMUS core: they depend on
matplotlib / anyplotlib, which are plot extras. If another example needs the
same bipolar colorplot, this is the file to promote.

The delay-and-sum.com colors (#0085FF / white / #FF195E) are more saturated
than matplotlib ``bwr``. anyplotlib's colorbar draws a strip without numeric
ticks; ``enable_colorbar_ticks`` patches that renderer before the figure is
created.
"""

from __future__ import annotations

from anyplotlib.figure._figure import Figure as AplFigure
from matplotlib import colormaps as mpl_cmaps
from matplotlib.colors import LinearSegmentedColormap

DAS_BIPOLAR = "das_bipolar"
_TICK_MARK = "fastsimus-cb-ticks"


def register_das_bipolar() -> str:
    """Register the delay-and-sum diverging colormap. Idempotent."""
    if DAS_BIPOLAR not in mpl_cmaps:
        mpl_cmaps.register(LinearSegmentedColormap.from_list(DAS_BIPOLAR, ["#0085FF", "#FFFFFF", "#FF195E"]))
    return DAS_BIPOLAR


def enable_colorbar_ticks() -> bool:
    """Patch anyplotlib so 2-D colorbars show numeric ticks.

    Returns True if the running anyplotlib copy was patched (or already had
    this patch). Returns False if the renderer source no longer matches, in
    which case the stock unlabeled strip is left as-is.
    """
    esm = AplFigure._esm
    if _TICK_MARK in esm:
        return True

    width_old = """  function _cbWidth(st) {
    if (!st || !st.show_colorbar || st.is_rgb) return 0;
    const labelW = st.colorbar_label
      ? Math.round((st.colorbar_label_size || 10) + 8) : 0;
    return 16 + labelW;
  }"""
    width_new = f"""  function _cbWidth(st) {{
    if (!st || !st.show_colorbar || st.is_rgb) return 0;
    const labelW = st.colorbar_label
      ? Math.round((st.colorbar_label_size || 10) + 8) : 0;
    const tickW = 36; /* {_TICK_MARK} */
    return 16 + tickW + labelW;
  }}"""
    ticks_old = """    // display_min / display_max tick marks
    const dMin=st.display_min, dMax=st.display_max;
    const [hMin,hMax]=_rawBand(st);
    const vRange=(hMax-hMin)||1;
    function _vToY(v){return imgH-1-((v-hMin)/vRange)*(imgH-1);}
    ctx.strokeStyle='rgba(255,255,255,0.85)'; ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(0,_vToY(dMax));ctx.lineTo(cbStripW,_vToY(dMax));ctx.stroke();
    ctx.beginPath();ctx.moveTo(0,_vToY(dMin));ctx.lineTo(cbStripW,_vToY(dMin));ctx.stroke();"""
    ticks_new = f"""    // display_min / display_max tick marks + numeric labels
    const dMin=st.display_min, dMax=st.display_max;
    const [hMin,hMax]=_rawBand(st);
    const vRange=(hMax-hMin)||1;
    const tickW=36; /* {_TICK_MARK} */
    function _vToY(v){{return imgH-1-((v-hMin)/vRange)*(imgH-1);}}
    function _fmtCbTick(v){{
      if(!Number.isFinite(v)) return '';
      if(Math.abs(v) < 1e-9 * Math.max(1, Math.abs(hMax), Math.abs(hMin))) return '0';
      const a=Math.abs(v);
      if(a>=1e4 || (a>0 && a<0.01)) return v.toExponential(1).replace('e+','e');
      if(Math.abs(v-Math.round(v))<1e-6) return String(Math.round(v));
      return String(Math.round(v*10)/10);
    }}
    ctx.strokeStyle='rgba(255,255,255,0.85)'; ctx.lineWidth=1;
    ctx.fillStyle=theme.tickText;
    const _tickPx=st.tick_size||9;
    for(let i=0;i<5;i++){{
      const v=hMin+(hMax-hMin)*i/4;
      const y=_vToY(v);
      ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(cbStripW,y);ctx.stroke();
      ctx.textBaseline=(y<8)?'top':(y>imgH-8)?'bottom':'middle';
      _drawTex(ctx,_fmtCbTick(v),cbStripW+3,y,_tickPx,{{align:'left'}});
    }}"""
    label_old = """    // Colorbar label (rotated \u221290\u00b0, centred in the label gutter)
    if(cbLabel){
      ctx.save();
      ctx.translate(cbStripW + (cbW - cbStripW) / 2 + 1, imgH/2);
      ctx.rotate(-Math.PI/2);
      ctx.textBaseline='middle';
      ctx.fillStyle=theme.unitText;
      _drawTex(ctx,cbLabel,0,0,st.colorbar_label_size||10,{align:'center'});
      ctx.restore();
    }"""
    label_new = """    // Colorbar label (rotated \u221290\u00b0, centred in the unit gutter)
    if(cbLabel){
      ctx.save();
      ctx.translate(cbStripW + tickW + (cbW - cbStripW - tickW) / 2 + 1, imgH/2);
      ctx.rotate(-Math.PI/2);
      ctx.textBaseline='middle';
      ctx.fillStyle=theme.unitText;
      _drawTex(ctx,cbLabel,0,0,st.colorbar_label_size||10,{align:'center'});
      ctx.restore();
    }"""
    if width_old not in esm or ticks_old not in esm or label_old not in esm:
        return False
    AplFigure._esm = esm.replace(width_old, width_new).replace(ticks_old, ticks_new).replace(label_old, label_new)
    return True
