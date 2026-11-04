import { useRef, useState, type PointerEvent } from "react";

// Pan is in root SVG units; zoom is around its centre. Hit-testing uses the
// transformed image group's inverse CTM, shared by pixels and annotations.
export function useImageNavigation(
  width: number,
  height: number,
  flipX = false,
  flipY = false,
) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [turned, setTurned] = useState(false);
  const [dragButton, setDragButton] = useState<number | null>(null);
  const drag = useRef<{
    id: number;
    button: number;
    x: number;
    y: number;
  } | null>(null);
  const suppressClick = useRef(false);
  const clampZoom = (value: number) => Math.max(0.5, Math.min(10, value));
  function stop(e: PointerEvent<SVGSVGElement>) {
    if (drag.current?.id !== e.pointerId) return;
    drag.current = null;
    setDragButton(null);
    if (e.currentTarget.hasPointerCapture(e.pointerId))
      e.currentTarget.releasePointerCapture(e.pointerId);
  }
  return {
    zoom,
    zoomBy: (factor: number) => setZoom((z) => clampZoom(z * factor)),
    rotate: () => setTurned((v) => !v),
    reset: () => {
      setZoom(1);
      setPan({ x: 0, y: 0 });
      setTurned(false);
    },
    canClick: () => !suppressClick.current,
    transform: `translate(${pan.x + width / 2} ${pan.y + height / 2}) scale(${zoom}) translate(${-width / 2} ${-height / 2}) translate(${flipX !== turned ? width : 0} ${flipY !== turned ? height : 0}) scale(${flipX !== turned ? -1 : 1} ${flipY !== turned ? -1 : 1})`,
    handlers: {
      style: {
        cursor:
          dragButton === 1
            ? "grabbing"
            : dragButton === 2
              ? "ns-resize"
              : "crosshair",
        touchAction: "none",
      },
      onContextMenu: (e: React.MouseEvent) => e.preventDefault(),
      onPointerDown: (e: PointerEvent<SVGSVGElement>) => {
        if (drag.current) return;
        suppressClick.current = e.button !== 0;
        if (e.button !== 1 && e.button !== 2) return;
        e.preventDefault();
        drag.current = {
          id: e.pointerId,
          button: e.button,
          x: e.clientX,
          y: e.clientY,
        };
        setDragButton(e.button);
        e.currentTarget.setPointerCapture(e.pointerId);
      },
      onPointerMove: (e: PointerEvent<SVGSVGElement>) => {
        const previous = drag.current;
        if (!previous || previous.id !== e.pointerId) return;
        e.preventDefault();
        if (previous.button === 2)
          setZoom((z) => clampZoom(z - (e.clientY - previous.y) * 0.01));
        else {
          const matrix = e.currentTarget.getScreenCTM();
          if (matrix) {
            const inverse = matrix.inverse();
            const a = new DOMPoint(previous.x, previous.y).matrixTransform(
              inverse,
            );
            const b = new DOMPoint(e.clientX, e.clientY).matrixTransform(
              inverse,
            );
            setPan((p) => ({ x: p.x + b.x - a.x, y: p.y + b.y - a.y }));
          }
        }
        drag.current = { ...previous, x: e.clientX, y: e.clientY };
      },
      onPointerUp: stop,
      onPointerCancel: stop,
      onLostPointerCapture: stop,
      onAuxClick: (e: React.MouseEvent) => e.preventDefault(),
    },
  };
}
