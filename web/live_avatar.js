import { app } from "../../scripts/app.js";

const viewUrl = (saved) => `/view?filename=${encodeURIComponent(saved.filename)}&subfolder=${encodeURIComponent(saved.subfolder)}&type=${saved.type}`;

function showViewer(node, saved) {
  if (!node.liveAvatarFrame) {
    node.liveAvatarFrame = document.createElement("iframe");
    node.liveAvatarFrame.style.cssText = "width:100%;height:100%;border:0;border-radius:6px";
    node.liveAvatarFrame.allow = "microphone; camera";
    node.addDOMWidget("viewer", "live_avatar", node.liveAvatarFrame, { serialize: false });
  }
  node.liveAvatarFrame.src = `/extensions/ComfyUI-Live-Avatar/viewer/index.html?file=${encodeURIComponent(viewUrl(saved))}`;
}

function showDownload(node, saved) {
  if (!node.liveAvatarLink) {
    node.liveAvatarLink = document.createElement("a");
    node.liveAvatarLink.style.cssText = "display:block;padding:6px;color:#6ea8fe;font:12px sans-serif";
    node.addDOMWidget("download", "live_avatar_export", node.liveAvatarLink, { serialize: false });
  }
  node.liveAvatarLink.textContent = `Download ${saved.filename}`;
  node.liveAvatarLink.href = viewUrl(saved);
  node.liveAvatarLink.download = saved.filename;
}

// The last result lives in node.properties: it is saved with the workflow, so switching tabs or reloading brings it back.
function remember(nodeType, key, show, afterFirst) {
  const onExecuted = nodeType.prototype.onExecuted;
  nodeType.prototype.onExecuted = function (message) {
    onExecuted?.apply(this, arguments);
    const saved = message?.[key]?.[0];
    if (!saved) return;
    const first = !this.properties[key];
    this.properties[key] = saved;
    show(this, saved);
    if (first) afterFirst?.(this);
  };
  const onConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    onConfigure?.apply(this, arguments);
    if (this.properties?.[key]) show(this, this.properties[key]);
  };
}

app.registerExtension({
  name: "LiveAvatar.Viewer",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "LiveAvatarSave") {
      remember(nodeType, "live_avatar", showViewer, (node) => node.setSize([Math.max(node.size[0], 420), Math.max(node.size[1], 620)]));
    } else if (nodeData.name === "LiveAvatarExport") {
      remember(nodeType, "live_avatar_export", showDownload);
    }
  },
});
