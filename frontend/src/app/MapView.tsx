"use client";
import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import proj4 from "proj4";
import { DetectionResult } from "./page";

// Define Stereo70 EPSG:3844 projection
proj4.defs("EPSG:3844", "+proj=sterea +lat_0=46 +lon_0=25 +k=0.99975 +x_0=500000 +y_0=500000 +ellps=krass +towgs84=2.329,-147.042,-92.08,0.309,-0.325,-0.497,5.69 +units=m +no_defs");

// Helper to convert Stereo70 to WGS84
const stereo70ToWgs84 = (x: number, y: number): [number, number] => {
  // proj4 transforms [x, y] to [lon, lat]
  const [lon, lat] = proj4("EPSG:3844", "EPSG:4326", [x, y]);
  return [lat, lon];
};

// Fix Leaflet's default marker icon issue in Next.js
const icon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  tooltipAnchor: [16, -28],
});

// Auto-adjust map boundaries to fit all markers dynamically
function MapBounds({ data }: { data: DetectionResult[] }) {
  const map = useMap();
  useEffect(() => {
    if (data && data.length > 0) {
      const bounds = L.latLngBounds(data.map(d => stereo70ToWgs84(d.x, d.y)));
      // Only fit bounds if they are valid
      if (bounds.isValid()) {
         map.fitBounds(bounds, { padding: [50, 50], maxZoom: 18 });
      }
    }
  }, [data, map]);
  return null;
}

export default function MapView({ data }: { data: DetectionResult[] }) {
  const center = data.length > 0 ? stereo70ToWgs84(data[0].x, data[0].y) : [45.7489, 21.2087];
  
  return (
    <MapContainer center={center as [number, number]} zoom={15} style={{ height: "100%", width: "100%", zIndex: 0 }} className="rounded-lg">
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
      <MapBounds data={data} />
      {data.map((item, idx) => {
        const [lat, lon] = stereo70ToWgs84(item.x, item.y);
        return (
        <Marker key={`${item.cam_key}_${item.image}_${idx}`} position={[lat, lon]} icon={icon}>
          <Popup>
            <div className="font-mono text-xs">
              <strong>Conf:</strong> {(item.conf * 100).toFixed(0)}% <br/>
              <strong>Image:</strong> {item.image} <br/>
              <strong>Cam:</strong> {item.cam_key} <br/>
              <strong>Z:</strong> {item.z.toFixed(2)}m
            </div>
          </Popup>
        </Marker>
        );
      })}
    </MapContainer>
  );
}
