interface Window {
  renderMathInElement: (
    element: HTMLElement, 
    options: {
      delimiters: Array<{left: string, right: string, display: boolean}>,
      throwOnError: boolean,
      [key: string]: any
    }
  ) => void;
}
