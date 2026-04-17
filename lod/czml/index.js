const Cesium = require("cesium");
const { readFileSync, writeFileSync } = require("fs");
const { join } = require("path");

function readJsonFile(filePath) {
  const data = readFileSync(filePath, "utf8");
  return JSON.parse(data);
}

const parseTextToJSON = (text) => {
  const regex = /chunk_(\d+_\d+): center\(([^)]+)\), axis\(([^)]+)\)/;
  const match = text.match(regex);

  if (match) {
    const [_, tile, center, axis] = match;
    const [centerX, centerY, centerZ] = center.split(" ").map(Number);
    const [axisX, axisY, axisZ] = axis.split(" ").map(Number);

    return {
      [tile]: {
        center: {
          x: centerX,
          y: centerY,
          z: centerZ,
        },
        axis: {
          x: axisX,
          y: axisY,
          z: axisZ,
        },
      },
    };
  } else {
    throw new Error("Text format is incorrect");
  }
};

const baseDir = process.argv[2];
const originFile = "gps_info.json";
const { longitude, latitude, altitude } = readJsonFile(
  join(baseDir, originFile)
);
const origin = Cesium.Cartesian3.fromDegrees(longitude, latitude, altitude);
const world = Cesium.Transforms.eastNorthUpToFixedFrame(origin);
const aabbfile = process.argv[3]
const data = readFileSync(join(baseDir, aabbfile), "utf8");
const lines = data.split("\n");
const aabbObj = {};

lines.forEach((line) => {
  if (line.trim()) {
    try {
      const parsed = parseTextToJSON(line);
      Object.assign(aabbObj, parsed);
    } catch (e) {
      console.error("Error parsing line:", line, e.message);
    }
  }
});

const czml = [
  {
    id: "document",
    name: "gs-aabb",
    version: "1.0",
  },
];

const tileIdArray = Object.keys(aabbObj);
for (const id of tileIdArray) {
  const { center, axis } = aabbObj[id];
  const centerWC = Cesium.Cartesian3.fromElements(center.x, center.y, center.z);
  Cesium.Matrix4.multiplyByPoint(world, centerWC, centerWC);
  const centerCarto = Cesium.Cartographic.fromCartesian(centerWC);
  const eyeOffset = Cesium.Cartesian3.normalize(
    centerWC,
    new Cesium.Cartesian3()
  );
  Cesium.Cartesian3.multiplyByScalar(eyeOffset, axis.z, eyeOffset);

  czml.push({
    id,
    name: id,
    position: {
      cartographicDegrees: [
        Cesium.Math.toDegrees(centerCarto.longitude),
        Cesium.Math.toDegrees(centerCarto.latitude),
        centerCarto.height,
      ],
    },
    label: {
      fillColor: {
        rgba: [255, 255, 0, 253],
      },
      font: "14pt Lucida Console",
      // horizontalOrigin: "CENTER",
      // eyeOffset: {
      //   cartesian: [eyeOffset.x, eyeOffset.y, eyeOffset.z],
      // },
      style: "FILL",
      text: id,
      showBackground: false,
    },
    box: {
      dimensions: {
        cartesian: [axis.x, axis.y, axis.z],
      },
      fill: false,
      outline: true,
      outlineColor: {
        rgba: [255, 255, 0, 255],
      },
    },
  });
}

const czmlString = JSON.stringify(czml, null, 2);
const outFile = `${aabbfile.replace(/\.txt$/, '.czml')}`;
writeFileSync(join(baseDir, outFile), czmlString, "utf8", (err) => {
  if (err) {
    console.error("Error writing to file:", err);
  } else {
    console.log("JSON data has been written to output.json");
  }
});
