
export function defineWorkerOnMessage<T>(handler: (data: T, ports: readonly MessagePort[]) => void) {
    return (event: MessageEvent) => {
        handler(event.data as T, event.ports);
    };
}