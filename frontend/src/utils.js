export function strNumber(num, unit='') {
  if (num > 1e14) {
    return `${(num / 1e12).toFixed(0)}T${unit}`;
  } else if (num > 1e12) {
    return `${(num / 1e12).toFixed(2)}T${unit}`;
  } else if (num > 1e11) {
    return `${(num / 1e9).toFixed(0)}G${unit}`;
  } else if (num > 1e9) {
    return `${(num / 1e9).toFixed(2)}G${unit}`;
  } else if (num > 1e8) {
    return `${(num / 1e6).toFixed(0)}M${unit}`;
  } else if (num > 1e6) {
    return `${(num / 1e6).toFixed(2)}M${unit}`;
  } else if (num > 1e5) {
    return `${(num / 1e3).toFixed(0)}K${unit}`;
  } else if (num > 1e3) {
    return `${(num / 1e3).toFixed(2)}K${unit}`;
  } else if (num >= 1) {
    return `${num.toFixed(2)}${unit}`;
  } else {
    return `${num.toFixed(2)}${unit}`;
  }
}

export function strNumberTime(num) {
  if (num >= 1) {
    return `${num.toFixed(2)}s`;
  } else if (num > 1e-3) {
    return `${(num * 1e3).toFixed(2)}ms`;
  } else if (num > 1e-6) {
    return `${(num * 1e6).toFixed(2)}us`;
  } else if (num > 1e-9) {
    return `${(num * 1e9).toFixed(2)}ns`;
  } else {
    return `${num.toFixed(0)}s`;
  }
}